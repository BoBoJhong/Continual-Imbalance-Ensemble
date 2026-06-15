"""
Phase 4 – Batch-level Concept Drift Detectors
==============================================
三種輕量偵測器，接收每個時間步（年份）的純量誤差信號（0–1），
不依賴任何外部串流學習套件（純 NumPy 實作）。

誤差信號建議使用 1 - AUC，對類別不平衡資料更穩健。

實作參考：
  ADWIN  : Bifet & Gavalda (2007) — Learning from Time-Changing Data with Adaptive Windowing
  DDM    : Gama et al. (2004)     — Learning with Drift Detection
  PHT    : Page (1954) / Hinkley  — Cumulative sum test
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ADWIN
# ─────────────────────────────────────────────────────────────────────────────
class ADWIN:
    """
    Adaptive Windowing（ADWIN）— 適用於實例級大量資料串流的高效版本。

    優化策略：
    1. 最大視窗上限 max_window（預設 2000）：防止 O(n²) 爆炸
    2. 每 check_every 筆才做分割檢查（預設 50）：降低分割頻率
    3. 分割點稀疏採樣：僅檢查指數間隔的分割點（log2(n) 個），
       保留 Hoeffding bound 的統計意義同時大幅降低計算量

    當視窗超過 max_window 時，自動丟棄最舊的資料（滑動視窗）。

    參數
    ----
    delta       : float  顯著性水準（越小 → 越保守）
    max_window  : int    視窗最大長度（控制記憶體與速度）
    check_every : int    每多少筆資料做一次分割檢查
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 2000,
        check_every: int = 50,
    ):
        self.delta = delta
        self.max_window = max_window
        self.check_every = check_every
        self._window: list[float] = []
        self._update_count = 0
        self.drift_detected = False
        self.n_detections = 0

    def update(self, value: float) -> bool:
        self._window.append(float(value))
        self._update_count += 1
        self.drift_detected = False

        # 超過最大視窗：丟棄最舊資料（近似滑動視窗）
        if len(self._window) > self.max_window:
            self._window = self._window[-self.max_window:]

        n = len(self._window)
        if n < 10 or (self._update_count % self.check_every) != 0:
            return False

        W = np.array(self._window)
        # 指數間隔分割點：只檢查 log2(n) 個位置，保留統計效力
        splits = sorted(set(
            int(n * f)
            for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ))

        for split in splits:
            if split < 5 or split > n - 5:
                continue
            W0, W1 = W[:split], W[split:]
            n0, n1 = len(W0), len(W1)
            bound = np.sqrt(
                (1.0 / (2 * n0) + 1.0 / (2 * n1)) * np.log(4.0 * n / self.delta)
            )
            if abs(W0.mean() - W1.mean()) >= bound:
                self._window = list(W[split:])
                self.drift_detected = True
                self.n_detections += 1
                self._update_count = 0
                return True
        return False

    def reset(self):
        self._window.clear()
        self._update_count = 0
        self.drift_detected = False

    @property
    def window_size(self) -> int:
        return len(self._window)


# ─────────────────────────────────────────────────────────────────────────────
# DDM
# ─────────────────────────────────────────────────────────────────────────────
class DDM:
    """
    Drift Detection Method（DDM）— Gama et al. (2004)。

    追蹤滾動誤差率 p 與其標準差 s。
    當 p + s 超過最佳值（p_min + s_min）的 drift_level 倍時，偵測到漂移。

    參數
    ----
    warning_level : float   p + s >= p_min + warning_level * s_min → 警告
    drift_level   : float   p + s >= p_min + drift_level   * s_min → 漂移
    min_samples   : int     累積至少 min_samples 個樣本才開始偵測
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 5,
    ):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self._reset()
        self.n_detections = 0

    def _reset(self):
        self._n = 0
        self._p = 1.0
        self._s = 1.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self.drift_detected = False
        self.warning_detected = False

    def update(self, value: float) -> bool:
        self._n += 1
        # 線上均值更新（Welford）
        self._p += (float(value) - self._p) / self._n
        self._s = np.sqrt(max(self._p * (1.0 - self._p) / self._n, 1e-12))
        self.drift_detected = False
        self.warning_detected = False

        if self._n < self.min_samples:
            return False

        ps = self._p + self._s
        if ps <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s

        if ps >= self._p_min + self.drift_level * self._s_min:
            self.drift_detected = True
            self.n_detections += 1
            self._reset()
            return True

        if ps >= self._p_min + self.warning_level * self._s_min:
            self.warning_detected = True

        return False

    def reset(self):
        self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# Page-Hinkley Test
# ─────────────────────────────────────────────────────────────────────────────
class PageHinkley:
    """
    Page-Hinkley Test（PHT）— 偵測均值向上漂移。

    支援兩種模式：
    - reference_mean=None：使用滾動均值（適合長序列，但對短序列不敏感）
    - reference_mean=float：使用固定基準均值（建議：以 burn-in 期計算，對年份級資料更有效）

    累積量 = Σ(x_t - ref_mean - delta)，從最小值起飛超過 threshold 時觸發。

    參數
    ----
    threshold      : float   觸發閾值 λ（越小 → 越靈敏）
    delta          : float   允許的最小漂移量（防止雜訊觸發）
    reference_mean : float|None  固定基準均值；None 表示使用滾動均值
    """

    def __init__(
        self,
        threshold: float = 5.0,
        delta: float = 0.005,
        reference_mean: float | None = None,
    ):
        self.threshold = threshold
        self.delta = delta
        self.reference_mean = reference_mean   # None → 滾動；float → 固定
        self._reset()
        self.n_detections = 0

    def _reset(self):
        self._n = 0
        self._rolling_mean = 0.0
        self._ph = 0.0
        self._ph_min = 0.0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self._n += 1
        self._rolling_mean += (float(value) - self._rolling_mean) / self._n
        ref = self.reference_mean if self.reference_mean is not None else self._rolling_mean
        self._ph += value - ref - self.delta
        self._ph_min = min(self._ph_min, self._ph)
        self.drift_detected = False

        if self._n < 5:
            return False

        if (self._ph - self._ph_min) > self.threshold:
            self.drift_detected = True
            self.n_detections += 1
            self._reset()
            return True
        return False

    def reset(self):
        self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# 工廠
# ─────────────────────────────────────────────────────────────────────────────
def make_detectors(
    pht_reference_mean: float | None = None,
) -> dict[str, ADWIN | DDM | PageHinkley]:
    """
    回傳 Phase 4 使用的三個偵測器實例（各自獨立）。

    設計給「個別實例級」誤差串流（0/1 per prediction）：
    - 每年約 5,000 筆，Hoeffding bound 才能收緊至 ~0.02
    - ADWIN delta=0.002（標準設定，n 夠大時即可偵測）
    - DDM min_samples=30（約 1/6 年的資料量即開始偵測）
    - PHT 使用 burn-in 固定基準均值（比滾動均值更敏感）

    參數
    ----
    pht_reference_mean : burn-in 期預測的平均誤差率；
                         若傳入 None 則使用滾動均值（不建議用於短序列）
    """
    return {
        "ADWIN": ADWIN(delta=0.002),
        "DDM":   DDM(warning_level=2.0, drift_level=3.0, min_samples=30),
        "PHT":   PageHinkley(
            threshold=50.0,
            delta=0.005,
            reference_mean=pht_reference_mean,
        ),
    }
