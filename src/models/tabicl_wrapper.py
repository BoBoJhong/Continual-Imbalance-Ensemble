"""TabICL (Tabular In-Context Learning) wrapper for binary classification.

This project uses the name **TabICL** to represent an in-context learning style
model for tabular classification. The implementation here is based on TabPFN
(Hollmann et al., 2022), which is a pretrained transformer for tabular data.

Interface is aligned with other wrappers in `src/models`:
  - fit(X_train, y_train)
  - predict(X) -> np.ndarray
  - predict_proba(X) -> np.ndarray  (positive-class probability, shape: (n,))

Notes
-----
- TabPFN has practical limits on the size of the context (training) set.
  This wrapper supports `max_train_samples` to cap the context size.
- This wrapper assumes *all features are numeric* (consistent with other
  Phase-1 baselines after preprocessing / scaling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SubsampleStrategy = Literal["stratified", "random"]


@dataclass(frozen=True)
class TabICLContextConfig:
    max_train_samples: int = 1024
    subsample: SubsampleStrategy = "stratified"


class TabICLWrapper:
    """TabICL binary classification wrapper (TabPFN-based)."""

    def __init__(
        self,
        name: str = "tabicl",
        *,
        device: str = "auto",
        n_ensemble_configurations: int = 16,
        seed: int = 42,
        max_train_samples: int = 1024,
        subsample: SubsampleStrategy = "stratified",
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyTorch 未安裝。請執行：pip install torch") from exc

        try:
            _ = self._import_tabpfn_classifier()
        except ImportError as exc:
            raise ImportError(
                "tabpfn 未安裝。請執行：pip install tabpfn （或加入 requirements.txt）"
            ) from exc

        self.name = str(name)
        self.device = str(device)
        self.n_ensemble_configurations = int(n_ensemble_configurations)
        self.seed = int(seed)
        self.context = TabICLContextConfig(
            max_train_samples=int(max_train_samples),
            subsample=str(subsample),
        )

        self._model = None

    @staticmethod
    def _import_tabpfn_classifier():
        """Import TabPFNClassifier with fallback for older tabpfn layouts."""
        try:
            from tabpfn import TabPFNClassifier

            return TabPFNClassifier
        except Exception:
            # Older versions expose the class under scripts/
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

            return TabPFNClassifier

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        d = str(device).lower().strip()
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if d in {"cpu", "cuda"}:
            if d == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' 但 torch.cuda.is_available() 為 False；"
                    "請安裝 CUDA 版 PyTorch 或改用 --device auto/cpu。"
                )
            return d
        raise ValueError(f"Unsupported device: {device}. Use 'auto', 'cuda', or 'cpu'.")

    @staticmethod
    def _to_numpy_X(X) -> np.ndarray:
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        return np.asarray(Xv, dtype=np.float32)

    @staticmethod
    def _to_numpy_y(y) -> np.ndarray:
        yv = y.values if hasattr(y, "values") else np.asarray(y)
        return np.asarray(yv, dtype=int).ravel()

    def _subsample_context(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(X.shape[0])
        cap = int(self.context.max_train_samples)
        if cap <= 0 or n <= cap:
            return X, y

        rng = np.random.default_rng(self.seed)

        if self.context.subsample == "random" or len(np.unique(y)) < 2:
            idx = rng.choice(n, size=cap, replace=False)
            idx = np.sort(idx)
            return X[idx], y[idx]

        # stratified
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        if len(idx_pos) == 0 or len(idx_neg) == 0:
            idx = rng.choice(n, size=cap, replace=False)
            idx = np.sort(idx)
            return X[idx], y[idx]

        # keep approximately the same ratio
        p = len(idx_pos) / n
        n_pos = int(round(cap * p))
        n_pos = min(max(n_pos, 1), cap - 1)
        n_neg = cap - n_pos

        take_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
        take_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)

        idx = np.concatenate([take_pos, take_neg])
        if len(idx) < cap:
            # fill the remainder randomly without replacement
            rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
            fill = rng.choice(rest, size=cap - len(idx), replace=False)
            idx = np.concatenate([idx, fill])

        idx = np.sort(idx)
        return X[idx], y[idx]

    def fit(self, X_train, y_train, **kwargs):
        """Fit TabPFN on the (possibly subsampled) context set."""
        import torch

        TabPFNClassifier = self._import_tabpfn_classifier()

        X = self._to_numpy_X(X_train)
        y = self._to_numpy_y(y_train)

        if X.shape[0] < 2:
            raise ValueError("TabICLWrapper requires at least 2 training samples")

        # Set seeds for deterministic subsampling & model behavior (best-effort).
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        Xc, yc = self._subsample_context(X, y)

        device = self._resolve_device(self.device)

        # TabPFN uses a slightly unusual parameter naming (`N_ensemble_configurations`).
        # Keep this robust by only passing the core expected args.
        # TabPFN uses slightly different parameter names across versions.
        try:
            self._model = TabPFNClassifier(
                device=device,
                N_ensemble_configurations=int(self.n_ensemble_configurations),
            )
        except TypeError:
            self._model = TabPFNClassifier(
                device=device,
                n_ensemble_configurations=int(self.n_ensemble_configurations),
            )

        self._model.fit(Xc, yc)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not trained yet")
        Xn = self._to_numpy_X(X)
        proba = self._model.predict_proba(Xn)
        # Expect shape (n, 2)
        return np.asarray(proba)[:, 1]

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= float(threshold)).astype(int)
