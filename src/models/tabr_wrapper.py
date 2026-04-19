"""
TabR wrapper for supervised tabular binary classification.

Paper: Gorishniy et al. (ICLR 2024)
       "TabR: Tabular Deep Learning Meets Nearest Neighbors"

This wrapper keeps the same high-level interface as the other model wrappers:
  fit(X_train, y_train, X_val=None, y_val=None)
  predict(X) -> np.ndarray
  predict_proba(X) -> np.ndarray  (positive-class probability, shape: (n,))
"""
from __future__ import annotations

import numpy as np


class TabRWrapper:
    """TabR binary classification wrapper with sklearn-like workflow."""

    def __init__(
        self,
        name: str = "tabr",
        cat_indices: list[int] | None = None,
        cat_cardinalities: list[int] | None = None,
        d_main: int = 96,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 2,
        predictor_n_blocks: int = 2,
        context_size: int = 96,
        context_sample_size: int | None = None,
        context_dropout: float = 0.0,
        dropout0: float = 0.2,
        dropout1: float = 0.2,
        normalization: str = "LayerNorm",
        activation: str = "ReLU",
        memory_efficient: bool = False,
        candidate_encoding_batch_size: int | None = None,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        seed: int = 42,
        device: str = "auto",
        predict_batch_size: int = 4096,
        val_batch_size: int = 8192,
        num_workers: int = 0,
        verbose: int = 0,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyTorch 未安裝。請執行：pip install torch") from exc
        try:
            from pytorch_tabr import TabRClassifier  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pytorch-tabr 未安裝。請先安裝可用的 TabR 套件與對應 faiss 後端，"
                "例如：pip install pytorch-tabr faiss-cpu"
            ) from exc

        self.name = name
        self.cat_indices = list(cat_indices or [])
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.d_main = int(d_main)
        self.d_multiplier = float(d_multiplier)
        self.encoder_n_blocks = int(encoder_n_blocks)
        self.predictor_n_blocks = int(predictor_n_blocks)
        self.context_size = int(context_size)
        self.context_sample_size = None if context_sample_size is None else int(context_sample_size)
        self.context_dropout = float(context_dropout)
        self.dropout0 = float(dropout0)
        self.dropout1 = float(dropout1)
        self.normalization = str(normalization)
        self.activation = str(activation)
        self.memory_efficient = bool(memory_efficient)
        self.candidate_encoding_batch_size = (
            None
            if candidate_encoding_batch_size is None
            else int(candidate_encoding_batch_size)
        )
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = str(device)
        self.predict_batch_size = int(predict_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.verbose = int(verbose)
        self._model = None

    @staticmethod
    def _resolve_training_device(device: str) -> str:
        import torch

        d = str(device).lower().strip()
        if d == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if d == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' 但 torch.cuda.is_available() 為 False；"
                    "請安裝 CUDA 版 PyTorch 或改用 --device auto/cpu。"
                )
            return "cuda"
        if d == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("device='mps' 但 MPS 不可用。")
            return "mps"
        if d == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported device: {device}. Use 'auto', 'cuda', 'cpu', or 'mps'.")

    @staticmethod
    def _as_float_array(X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return np.asarray(X_arr, dtype=np.float32)

    @staticmethod
    def _as_label_array(y):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        return np.asarray(y_arr).astype(np.int64).ravel()

    def _build_model(self):
        from pytorch_tabr import TabRClassifier

        return TabRClassifier(
            cat_indices=self.cat_indices,
            cat_cardinalities=self.cat_cardinalities,
            d_main=self.d_main,
            d_multiplier=self.d_multiplier,
            encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks,
            context_size=self.context_size,
            context_sample_size=self.context_sample_size,
            context_dropout=self.context_dropout,
            dropout0=self.dropout0,
            dropout1=self.dropout1,
            normalization=self.normalization,
            activation=self.activation,
            memory_efficient=self.memory_efficient,
            candidate_encoding_batch_size=self.candidate_encoding_batch_size,
            optimizer_params={"lr": self.lr, "weight_decay": self.weight_decay},
            device_name=self._resolve_training_device(self.device),
            seed=self.seed,
            verbose=self.verbose,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from sklearn.model_selection import train_test_split

        X = self._as_float_array(X_train)
        y = self._as_label_array(y_train)

        if X_val is not None and y_val is not None:
            Xv = self._as_float_array(X_val)
            yv = self._as_label_array(y_val)
        else:
            try:
                X, Xv, y, yv = train_test_split(
                    X,
                    y,
                    test_size=0.1,
                    random_state=self.seed,
                    stratify=y,
                )
            except ValueError:
                X, Xv, y, yv = train_test_split(
                    X,
                    y,
                    test_size=0.1,
                    random_state=self.seed,
                )

        model = self._build_model()
        model.fit(
            X,
            y,
            eval_set=[(Xv, yv)],
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=max(1, self.batch_size),
            num_workers=max(0, self.num_workers),
            drop_last=False,
            pin_memory=self._resolve_training_device(self.device) != "cpu",
            warm_start=bool(kwargs.get("continue_training", False)),
        )
        self._model = model
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not trained yet")

        X_arr = self._as_float_array(X)
        old_batch_size = getattr(self._model, "batch_size", None)
        if old_batch_size is not None:
            self._model.batch_size = max(1, self.predict_batch_size)
        try:
            proba = np.asarray(self._model.predict_proba(X_arr))
        finally:
            if old_batch_size is not None:
                self._model.batch_size = old_batch_size

        if proba.ndim == 1:
            return proba.astype(np.float32)
        if proba.shape[1] == 1:
            return proba[:, 0].astype(np.float32)
        return proba[:, 1].astype(np.float32)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
