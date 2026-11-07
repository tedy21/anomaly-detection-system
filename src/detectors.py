"""Anomaly detectors. Every detector follows the same interface:

    detector.fit(X_train)        # X_train: (n_steps, n_features), normal data
    scores = detector.score(X)   # one anomaly score per time step; higher = more abnormal

Thresholding is done outside the detector (see experiments/run_baseline.py)
so that all methods are compared with the same rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ZScoreDetector:
    """Statistical baseline.

    mode="global"  : z-score against the mean/std of the training data.
    mode="rolling" : z-score against the mean/std of the previous `window`
                     steps (adapts to slow drift, never looks at the future).

    The score of a time step is the largest |z| over all features.
    `std_floor` stops near-constant features from producing huge scores
    (SMD features are scaled to [0, 1], so 0.01 is a small but safe floor).
    """

    def __init__(self, mode: str = "global", window: int = 100, std_floor: float = 1e-2):
        if mode not in ("global", "rolling"):
            raise ValueError("mode must be 'global' or 'rolling'")
        self.mode = mode
        self.window = window
        self.std_floor = std_floor
        self.name = f"zscore_{mode}" + (f"_w{window}" if mode == "rolling" else "")

    def fit(self, X_train: np.ndarray) -> "ZScoreDetector":
        X_train = np.asarray(X_train, dtype=float)
        self.mean_ = X_train.mean(axis=0)
        self.std_ = np.maximum(X_train.std(axis=0), self.std_floor)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mode == "global":
            z = np.abs((X - self.mean_) / self.std_)
            return z.max(axis=1)

        df = pd.DataFrame(X)
        min_periods = max(10, self.window // 10)
        roll = df.rolling(self.window, min_periods=min_periods)
        mean = roll.mean().shift(1)  # shift(1): use only the past
        std = roll.std().shift(1).clip(lower=self.std_floor)
        z = (df - mean).abs() / std
        return z.max(axis=1).fillna(0.0).to_numpy()
