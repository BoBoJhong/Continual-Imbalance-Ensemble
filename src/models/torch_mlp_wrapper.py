"""PyTorch MLP for tabular binary classification（Phase 1 深度學習基線）。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TabularMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchTabularMLPWrapper:
    """
    監督式深度基線：多層感知器（PyTorch）。

    介面與 XGBoostWrapper 一致：
      fit(X_train, y_train) -> self
      predict_proba(X) -> np.ndarray 正類機率，shape (n,)
    """

    def __init__(
        self,
        name: str = "torch_mlp",
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 80,
        patience: int = 12,
        batch_size: int = 2048,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.name = name
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _X_to_float32(X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return np.asarray(X_arr, dtype=np.float32)

    @staticmethod
    def _y_to_float64(y) -> np.ndarray:
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        return np.asarray(y_arr, dtype=np.float64).ravel()

    def _pos_weight(self, y: np.ndarray) -> float:
        y_int = y.astype(int)
        n_pos = int((y_int == 1).sum())
        n_neg = int((y_int == 0).sum())
        if n_pos == 0:
            return 1.0
        return float(n_neg) / float(n_pos)

    def fit(self, X_train, y_train, **kwargs):
        _set_seed(self.seed)
        X = self._X_to_float32(X_train)
        y = self._y_to_float64(y_train)
        n = len(X)
        if n < 2:
            raise ValueError("TorchTabularMLPWrapper requires at least 2 training samples")

        from sklearn.model_selection import train_test_split

        vf = min(max(self.val_fraction, 0.05), 0.49)
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=vf, random_state=self.seed, stratify=y.astype(int)
            )
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=vf, random_state=self.seed
            )

        pos_w = self._pos_weight(y_tr)
        pw_tensor = torch.tensor([pos_w], device=self._device, dtype=torch.float32)

        n_features = int(X_tr.shape[1])
        self._model = _TabularMLP(n_features, self.hidden, self.dropout).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        bs = max(1, min(self.batch_size, len(X_tr)))
        train_ds = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr.astype(np.float32)),
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

        X_va_t = torch.from_numpy(X_va).to(self._device)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(self._device)

        best_state = None
        best_loss = float("inf")
        bad_epochs = 0

        for _ in range(self.max_epochs):
            self._model.train()
            for xb, yb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                opt.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_va_t)
                val_loss = float(loss_fn(val_logits, y_va_t).item())

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def continue_fit(
        self,
        X_train,
        y_train,
        lr_factor: float = 0.25,
        max_epochs: int | None = None,
        pos_weight_override: float | None = None,
        **kwargs,
    ):
        """
        在既有權重上接續訓練（對齊 XGBoost 以 xgb_model 接續微調 New 資料）。
        若給 pos_weight_override（例如對齊 XGB 以整段 y_r2 算的 scale_pos_weight），
        則 BCE 使用該值；否則用內部 train fold 的類別比估算。
        """
        if self._model is None:
            raise ValueError("continue_fit requires fit() first")

        X = self._X_to_float32(X_train)
        y = self._y_to_float64(y_train)
        n = len(X)
        if n < 2:
            raise ValueError("continue_fit requires at least 2 training samples")

        from sklearn.model_selection import train_test_split

        vf = min(max(self.val_fraction, 0.05), 0.49)
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=vf, random_state=self.seed + 1, stratify=y.astype(int)
            )
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=vf, random_state=self.seed + 1
            )

        if pos_weight_override is not None:
            pos_w = float(pos_weight_override)
        else:
            pos_w = self._pos_weight(y_tr)
        pw_tensor = torch.tensor([pos_w], device=self._device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        me = self.max_epochs if max_epochs is None else int(max_epochs)
        lr = float(self.lr) * float(lr_factor)
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=self.weight_decay)

        bs = max(1, min(self.batch_size, len(X_tr)))
        train_ds = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr.astype(np.float32)),
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

        X_va_t = torch.from_numpy(X_va).to(self._device)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(self._device)

        best_state = None
        best_loss = float("inf")
        bad_epochs = 0

        for _ in range(me):
            self._model.train()
            for xb, yb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                opt.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_va_t)
                val_loss = float(loss_fn(val_logits, y_va_t).item())

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not trained yet")
        X_arr = self._X_to_float32(X)
        self._model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_arr).to(self._device)
            logits = self._model(t)
            proba = torch.sigmoid(logits).cpu().numpy()
        return np.asarray(proba, dtype=np.float64)
