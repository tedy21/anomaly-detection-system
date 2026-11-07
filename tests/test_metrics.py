import numpy as np
import pytest

from src.detectors import ZScoreDetector
from src.metrics import (
    event_based,
    evaluate,
    point_adjust,
    point_adjusted,
    pointwise,
    segments,
)


def test_segments():
    assert segments([0, 1, 1, 0, 1, 0, 0, 1, 1]) == [(1, 2), (4, 4), (7, 8)]
    assert segments([0, 0, 0]) == []
    assert segments([1, 1]) == [(0, 1)]


def test_pointwise_perfect_and_empty():
    y = np.array([0, 1, 1, 0])
    assert pointwise(y, y)["f1"] == 1.0
    assert pointwise(y, np.zeros(4))["f1"] == 0.0


def test_point_adjust_fills_segment():
    y_true = np.array([0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    assert point_adjust(y_true, y_pred).tolist() == [0, 1, 1, 1, 0, 1]


def test_point_adjusted_inflates_score():
    # One hit in a long anomaly: point-wise recall is low, point-adjusted is 1.
    y_true = np.array([0] + [1] * 10 + [0])
    y_pred = np.array([0, 1] + [0] * 10)
    assert pointwise(y_true, y_pred)["recall"] == pytest.approx(0.1)
    assert point_adjusted(y_true, y_pred)["recall"] == 1.0


def test_event_based_counts_events_and_alarms():
    y_true = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1])  # hits event 1; two false alarms
    r = event_based(y_true, y_pred)
    assert r["recall"] == pytest.approx(0.5)  # 1 of 2 events
    assert r["precision"] == pytest.approx(1 / 3)  # 1 of 3 alarms


def test_evaluate_keys_and_single_class_auc():
    y = np.zeros(10, dtype=int)
    out = evaluate(y, np.random.rand(10), 0.5)
    assert np.isnan(out["auroc"])
    for key in ("pw_f1", "pa_f1", "ev_f1", "auprc"):
        assert key in out


def test_zscore_detects_spike_and_uses_only_past():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 3))
    X_spike = X.copy()
    X_spike[300:305, 0] += 10
    for mode in ("global", "rolling"):
        det = ZScoreDetector(mode).fit(X)
        s = det.score(X_spike)
        assert s[300] > 5  # spike start is caught by both modes
        if mode == "global":
            assert s[300:305].min() > 5
        # rolling mode: the window absorbs a long spike, so later points score lower
        assert s[:300].max() < 6
    # rolling scores up to t must not change if the future changes
    det = ZScoreDetector("rolling").fit(X)
    a = det.score(X)[:200]
    X_future = X.copy()
    X_future[250:] += 100
    b = det.score(X_future)[:200]
    np.testing.assert_allclose(a, b)
