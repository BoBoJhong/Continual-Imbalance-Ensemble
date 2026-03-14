"""
TabNet 模型 wrapper
=====================
論文：Arik & Pfister (AAAI 2021) - TabNet: Attentive Interpretable Tabular Learning
實作：pytorch-tabnet (pip install pytorch-tabnet)

介面與 MLPWrapper、XGBoostWrapper 完全一致：
  fit(X_train, y_train)
  predict(X) -> np.ndarray
  predict_proba(X) -> np.ndarray  (正類機率，shape: (n,))
"""
import numpy as np


class TabNetWrapper:
    """
    TabNet 分類器 wrapper。

    不平衡處理：
      - 訓練前由 ImbalanceSampler 負責（採樣後餵入此模型）
      - 可透過 class_weight 參數額外加權（預設 None 不加權）

    重要超參數：
      n_d, n_a   : 決策步驟的特徵寬度（預設 32/32，值越大模型越複雜）
      n_steps    : Sequential Attention 的步驟數（預設 5）
      gamma      : 先前步驟特徵的抑制係數（預設 1.3）
      max_epochs : 最大訓練 epoch 數
      patience   : early stopping 的等待 epoch 數
    """

    def __init__(
        self,
        name: str = "tabnet",
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
    ):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            raise ImportError(
                "pytorch-tabnet 未安裝。請執行：pip install pytorch-tabnet"
            )

        self.name = name
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            seed=seed,
            verbose=verbose,
            optimizer_fn=__import__("torch").optim.Adam,
            optimizer_params={"lr": 2e-3},
            scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 50, "gamma": 0.9},
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        訓練 TabNet。
        若未提供 X_val / y_val，自動從訓練集分割 10% 作為 early stopping 用。
        """
        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train, dtype=np.float32)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        X = X.astype(np.float32)
        y = y.astype(int)

        if X_val is not None and y_val is not None:
            Xv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val, dtype=np.float32)
            yv = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)
            eval_set = [(Xv.astype(np.float32), yv.astype(int))]
        else:
            # 自動分割 10% 作為驗證集
            from sklearn.model_selection import train_test_split
            X, Xv, y, yv = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y if y.sum() > 1 else None)
            eval_set = [(Xv, yv)]

        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
        )
        return self

    def predict(self, X) -> np.ndarray:
        X = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        return self.model.predict(X.astype(np.float32))

    def predict_proba(self, X) -> np.ndarray:
        """回傳正類機率（shape: (n,)）"""
        X = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        proba = self.model.predict_proba(X.astype(np.float32))
        return proba[:, 1]
