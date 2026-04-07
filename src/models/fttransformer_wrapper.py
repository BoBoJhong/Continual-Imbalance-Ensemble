"""
FT-Transformer (Feature Tokenizer + Transformer) 模型 wrapper
================================================================
論文：Gorishniy et al. (NeurIPS 2021)
      "Revisiting Deep Learning Models for Tabular Data"
實作：rtdl (pip install rtdl)  → rtdl.FTTransformer

介面與 MLPWrapper、XGBoostWrapper 完全一致：
  fit(X_train, y_train)
  predict(X) -> np.ndarray
  predict_proba(X) -> np.ndarray  (正類機率，shape: (n,))

訓練：預設在 CUDA 上啟用 mixed precision (AMP)；不平衡二元分類可設 pos_weight='auto'
（依訓練集 n_neg/n_pos）。推論時模型在 CPU，大批量 predict_proba 可自動分批。
"""
import os
import sys
import numpy as np


class FTTransformerWrapper:
    """
    FT-Transformer 分類器 wrapper。

    核心概念：
      每個數值特徵 → learnable linear embedding（Feature Tokenizer）
      → 拼上 [CLS] token
      → 標準 Transformer Encoder（多頭自注意力 + FFN）
      → 取 CLS 輸出 → Linear → 預測

    重要超參數：
      d_token        : 每個特徵的 embedding 維度（預設 64）
      n_blocks       : Transformer Encoder 的層數（預設 3）
      attention_n_heads  : 多頭注意力的 head 數（預設 8）
      ffn_d_hidden   : FFN 隱藏層維度（預設 d_token * 4/3）
      attention_dropout  : attention dropout（預設 0.2）
      ffn_dropout    : FFN dropout（預設 0.1）
      residual_dropout   : residual dropout（預設 0.0）
      max_epochs     : 最大訓練 epoch 數
      patience       : early stopping 等待 epoch 數
    """

    def __init__(
        self,
        name: str = "fttransformer",
        d_token: int = 64,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        seed: int = 42,
        device: str = "auto",
        pos_weight: str | float | None = "auto",
        use_amp: bool = True,
        val_batch_size: int = 8192,
        predict_batch_size: int = 4096,
    ):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch 未安裝。請執行：pip install torch")
        try:
            import rtdl  # noqa: F401
        except ImportError:
            raise ImportError("rtdl 未安裝。請執行：pip install rtdl")

        self.name = name
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.pos_weight = pos_weight
        self.use_amp = bool(use_amp)
        self.val_batch_size = int(val_batch_size)
        self.predict_batch_size = int(predict_batch_size)
        self._model = None
        self._n_features = None
        self._train_device: str | None = None

    def _build_model(self, n_features: int):
        import rtdl
        import torch
        torch.manual_seed(self.seed)
        ffn_d_hidden = int(self.d_token * 4 / 3)
        model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            d_out=1,  # binary classification
        )
        return model

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
            v = n_neg / n_pos
            return torch.tensor(v, dtype=torch.float32, device=device)
        return torch.tensor(float(pw), dtype=torch.float32, device=device)

    @staticmethod
    def _val_loss_batched(model, Xvt, yvt, criterion, device, batch_size: int) -> float:
        import torch

        model.eval()
        n = Xvt.shape[0]
        if n == 0:
            return float("inf")
        total, seen = 0.0, 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = Xvt[i : i + batch_size].to(device)
                yb = yvt[i : i + batch_size].to(device)
                logits = model(xb, None).squeeze(1)
                loss = criterion(logits, yb)
                bs = xb.shape[0]
                total += loss.item() * bs
                seen += bs
        return total / max(seen, 1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split

        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # 建立驗證集
        if X_val is not None and y_val is not None:
            Xv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val, dtype=np.float32)
            yv = y_val.values if hasattr(y_val, "values") else np.asarray(y_val, dtype=np.float32)
        else:
            try:
                X, Xv, y, yv = train_test_split(X, y, test_size=0.1, random_state=self.seed, stratify=y.astype(int))
            except ValueError:
                X, Xv, y, yv = train_test_split(X, y, test_size=0.1, random_state=self.seed)

        continue_training = bool(kwargs.get("continue_training", False))
        n_features = X.shape[1]

        if continue_training and self._model is not None and self._n_features == n_features:
            model = self._model
        else:
            self._n_features = n_features
            model = self._build_model(self._n_features)

        device = self._resolve_training_device(self.device)
        self._train_device = str(device)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.seed))

        use_amp = self.use_amp and device.type == "cuda"
        # MPS 上 AMP 行為因版本而異，僅在 CUDA 啟用
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pw_t = self._pos_weight_tensor(y, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t) if pw_t is not None else nn.BCEWithLogitsLoss()

        # 資料集留在 CPU；pin_memory 僅對 CPU tensor 有效，batch 再在迴圈內 .to(device)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        Xvt = torch.tensor(Xv, dtype=torch.float32)
        yvt = torch.tensor(yv, dtype=torch.float32)

        pin_memory = device.type == "cuda"
        nw = 0 if sys.platform.startswith("win") else min(4, (os.cpu_count() or 2) // 2)
        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(xb, None).squeeze(1)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb, None).squeeze(1)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

            val_loss = self._val_loss_batched(
                model, Xvt, yvt, criterion, device, self.val_batch_size
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
        """回傳正類機率（shape: (n,)）；大批次時分段 forward 省記憶體。"""
        import torch
        self._model.eval()
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        X_arr = X_arr.astype(np.float32, copy=False)
        n = X_arr.shape[0]
        bs = max(1, int(self.predict_batch_size))
        chunks = []
        with torch.no_grad():
            for i in range(0, n, bs):
                Xt = torch.tensor(X_arr[i : i + bs])
                logits = self._model(Xt, None).squeeze(1)
                chunks.append(torch.sigmoid(logits).numpy())
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
