"""
TabM wrapper for supervised tabular binary classification.

論文：Gorishniy et al. (ICLR 2025)
      "TabM: Advancing tabular deep learning with parameter-efficient ensembling"
實作：tabm (pip install tabm)

介面與其他 wrapper 對齊：
  fit(X_train, y_train)
  predict(X) -> np.ndarray
  predict_proba(X) -> np.ndarray  (正類機率，shape: (n,))
"""
from __future__ import annotations

import os
import sys

import numpy as np


class TabMWrapper:
    """
    TabM 二元分類 wrapper。

    設計重點：
      - 訓練時對 k 個 ensemble member 的 logits 分別計算 loss
      - 推論時先做 sigmoid，再對 k 個機率取平均
      - 可選擇為數值特徵加上 LinearReLU embeddings
    """

    def __init__(
        self,
        name: str = "tabm",
        arch_type: str = "tabm",
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        use_num_embeddings: bool = False,
        d_embedding: int = 8,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 3e-4,
        seed: int = 42,
        device: str = "auto",
        pos_weight: str | float | None = "auto",
        use_amp: bool = True,
        val_batch_size: int = 8192,
        predict_batch_size: int = 4096,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyTorch 未安裝。請執行：pip install torch") from exc
        try:
            import tabm  # noqa: F401
        except ImportError as exc:
            raise ImportError("tabm 未安裝。請執行：pip install tabm") from exc

        self.name = name
        self.arch_type = str(arch_type)
        self.k = int(k)
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.dropout = float(dropout)
        self.use_num_embeddings = bool(use_num_embeddings)
        self.d_embedding = int(d_embedding)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = device
        self.pos_weight = pos_weight
        self.use_amp = bool(use_amp)
        self.val_batch_size = int(val_batch_size)
        self.predict_batch_size = int(predict_batch_size)
        self._model = None
        self._n_features: int | None = None
        self._train_device: str | None = None

    def _build_model(self, n_features: int):
        import torch
        from tabm import TabM

        torch.manual_seed(self.seed)
        num_embeddings = None
        if self.use_num_embeddings:
            from rtdl_num_embeddings import LinearReLUEmbeddings

            num_embeddings = LinearReLUEmbeddings(n_features, self.d_embedding)

        return TabM.make(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            k=self.k,
            arch_type=self.arch_type,
            num_embeddings=num_embeddings,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
        )

    @staticmethod
    def _resolve_training_device(device: str):
        import torch

        d = str(device).lower().strip()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if d == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' 但 torch.cuda.is_available() 為 False；請安裝 CUDA 版 PyTorch 或改用 --device auto/cpu。"
                )
            return torch.device("cuda")
        if d == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("device='mps' 但 MPS 不可用（需 Apple Silicon + 支援的 PyTorch）。")
            return torch.device("mps")
        if d == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device: {device}. Use 'auto', 'cuda', 'cpu', or 'mps'.")

    def _pos_weight_tensor(self, y: np.ndarray, device):
        import torch

        pw = self.pos_weight
        if pw is None or pw == "none" or pw == 1.0:
            return None
        if pw == "auto":
            yb = np.asarray(y).astype(np.float64).ravel()
            n_pos = float(yb.sum())
            n_neg = float(len(yb) - n_pos)
            if n_pos < 1.0:
                return None
            return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
        return torch.tensor(float(pw), dtype=torch.float32, device=device)

    @staticmethod
    def _val_loss_batched(model, Xvt, yvt, criterion, device, batch_size: int) -> float:
        import torch

        model.eval()
        n = Xvt.shape[0]
        if n == 0:
            return float("inf")

        total = 0.0
        seen = 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = Xvt[i : i + batch_size].to(device)
                yb = yvt[i : i + batch_size].to(device)
                logits = model(xb).squeeze(-1)
                target = yb[:, None].expand(-1, logits.shape[1])
                loss = criterion(logits, target)
                bs = xb.shape[0]
                total += float(loss.item()) * bs
                seen += bs
        return total / max(seen, 1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        if X_val is not None and y_val is not None:
            Xv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
            yv = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)
            Xv = np.asarray(Xv, dtype=np.float32)
            yv = np.asarray(yv, dtype=np.float32).ravel()
        else:
            try:
                X, Xv, y, yv = train_test_split(
                    X,
                    y,
                    test_size=0.1,
                    random_state=self.seed,
                    stratify=y.astype(int),
                )
            except ValueError:
                X, Xv, y, yv = train_test_split(
                    X,
                    y,
                    test_size=0.1,
                    random_state=self.seed,
                )

        continue_training = bool(kwargs.get("continue_training", False))
        n_features = int(X.shape[1])
        if continue_training and self._model is not None and self._n_features == n_features:
            model = self._model
        else:
            self._n_features = n_features
            model = self._build_model(n_features)

        device = self._resolve_training_device(self.device)
        self._train_device = str(device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        use_amp = self.use_amp and device.type == "cuda"
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pw_t = self._pos_weight_tensor(y, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t) if pw_t is not None else nn.BCEWithLogitsLoss()

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        Xvt = torch.tensor(Xv, dtype=torch.float32)
        yvt = torch.tensor(yv, dtype=torch.float32)

        pin_memory = device.type == "cuda"
        nw = 0 if sys.platform.startswith("win") else min(4, (os.cpu_count() or 2) // 2)
        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=max(1, self.batch_size),
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=nw,
            persistent_workers=nw > 0,
        )
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(self.max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(xb).squeeze(-1)
                        target = yb[:, None].expand(-1, logits.shape[1])
                        loss = criterion(logits, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb).squeeze(-1)
                    target = yb[:, None].expand(-1, logits.shape[1])
                    loss = criterion(logits, target)
                    loss.backward()
                    optimizer.step()

            val_loss = self._val_loss_batched(
                model,
                Xvt,
                yvt,
                criterion,
                device,
                self.val_batch_size,
            )
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to("cpu")
        self._model = model
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        if self._model is None:
            raise ValueError("Model not trained yet")

        self._model.eval()
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        n = X_arr.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)

        bs = max(1, int(self.predict_batch_size))
        chunks = []
        with torch.no_grad():
            for i in range(0, n, bs):
                Xt = torch.tensor(X_arr[i : i + bs], dtype=torch.float32)
                logits = self._model(Xt).squeeze(-1)
                proba = torch.sigmoid(logits).mean(dim=1)
                chunks.append(proba.numpy())
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
