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
"""
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
        self._model = None
        self._n_features = None

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        Xt = torch.tensor(X, device=device)
        yt = torch.tensor(y, device=device)
        Xvt = torch.tensor(Xv, device=device)
        yvt = torch.tensor(yv, device=device)

        train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb, None).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # 驗證
            model.eval()
            with torch.no_grad():
                val_logits = model(Xvt, None).squeeze(1)
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
        """回傳正類機率（shape: (n,)）"""
        import torch
        self._model.eval()
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        Xt = torch.tensor(X_arr.astype(np.float32))
        with torch.no_grad():
            logits = self._model(Xt, None).squeeze(1)
            proba = torch.sigmoid(logits).numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
