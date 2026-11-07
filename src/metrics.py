"""Evaluation metrics for time-series anomaly detection.

Three ways to score the same predictions (research question 3):

* point-wise     : every time step counts equally (strict).
* point-adjusted : if any point inside a true anomaly segment is flagged,
                   the whole segment counts as detected (common in papers,
                   known to inflate scores).
* event-based    : each true anomaly segment is one event; each predicted
                   segment is one alarm.

All functions take 0/1 numpy arrays of equal length.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def segments(labels) -> list[tuple[int, int]]:
    """Return (start, end) inclusive index pairs for each run of 1s."""
    labels = np.asarray(labels).astype(int)
    if labels.size == 0:
        return []
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def pointwise(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return _prf(tp, fp, fn)


def point_adjust(y_true, y_pred) -> np.ndarray:
    """Expand predictions: a hit anywhere in a true segment fills the segment."""
    y_true = np.asarray(y_true).astype(int)
    adjusted = np.asarray(y_pred).astype(int).copy()
    for start, end in segments(y_true):
        if adjusted[start : end + 1].any():
            adjusted[start : end + 1] = 1
    return adjusted


def point_adjusted(y_true, y_pred) -> dict[str, float]:
    return pointwise(y_true, point_adjust(y_true, y_pred))


def event_based(y_true, y_pred) -> dict[str, float]:
    """Event recall = true events with any flagged point.
    Event precision = predicted alarms that overlap a true event."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    true_events = segments(y_true)
    alarms = segments(y_pred)
    detected = sum(1 for s, e in true_events if y_pred[s : e + 1].any())
    good_alarms = sum(1 for s, e in alarms if y_true[s : e + 1].any())
    recall = detected / len(true_events) if true_events else 0.0
    precision = good_alarms / len(alarms) if alarms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def threshold_free(y_true, scores) -> dict[str, float]:
    """AUROC and AUPRC; NaN if the labels contain only one class."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def evaluate(y_true, scores, threshold: float) -> dict[str, float]:
    """All metrics for one detector at one threshold, as a flat dict."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > threshold).astype(int)
    out: dict[str, float] = {"threshold": float(threshold)}
    for name, fn in (
        ("pw", pointwise),
        ("pa", point_adjusted),
        ("ev", event_based),
    ):
        for key, value in fn(y_true, y_pred).items():
            out[f"{name}_{key}"] = value
    out.update(threshold_free(y_true, scores))
    return out
