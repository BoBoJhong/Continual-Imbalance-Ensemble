"""
LSTM 模型 wrapper（tabular 特徵視為序列）
=============================================
將每筆樣本的數值特徵視為一段序列（seq_len = n_features, input_size = 1）。

介面與 MLPWrapper、XGBoostWrapper 完全一致：
  fit(X_train, y_train)
  predict(X) -> np.ndarray
  predict_proba(X) -> np.ndarray  (正類機率，shape: (n,))
"""
import numpy as np


class LSTMWrapper:
    """
    LSTM 分類器 wrapper。

    重要超參數：
      hidden_size : LSTM 隱藏維度
      num_layers  : LSTM 層數
      dropout     : LSTM dropout
      max_epochs  : 最大訓練 epoch 數
      patience    : early stopping 的等待 epoch 數
    """

    def __init__(
        self,
        name: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int = 42,
    ):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch 未安裝。請執行：pip install torch")

        self.name = name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._model = None
        self._n_features = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        class _LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.fc(last).squeeze(1)

        torch.manual_seed(self.seed)
        return _LSTMNet(1, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split

        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        if X_val is not None and y_val is not None:
            Xv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val, dtype=np.float32)
            yv = y_val.values if hasattr(y_val, "values") else np.asarray(y_val, dtype=np.float32)
        else:
            try:
                X, Xv, y, yv = train_test_split(
                    X, y, test_size=0.1, random_state=self.seed, stratify=y.astype(int)
                )
            except ValueError:
                X, Xv, y, yv = train_test_split(X, y, test_size=0.1, random_state=self.seed)

        self._n_features = X.shape[1]
        X = X.reshape(X.shape[0], self._n_features, 1)
        Xv = Xv.reshape(Xv.shape[0], self._n_features, 1)

        model = self._build_model(self._n_features)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        Xt = torch.tensor(X, device=device)
        yt = torch.tensor(y, device=device)
        Xvt = torch.tensor(Xv, device=device)
        yvt = torch.tensor(yv, device=device)

        train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for _ in range(self.max_epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(Xvt)
                val_loss = criterion(val_logits, yvt).item()

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
        self._model.eval()
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        X_arr = X_arr.astype(np.float32)
        X_arr = X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 1)
        Xt = torch.tensor(X_arr)
        with torch.no_grad():
            logits = self._model(Xt)
            proba = torch.sigmoid(logits).numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)