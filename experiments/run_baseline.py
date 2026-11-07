"""Week 1-2 experiment: z-score baselines scored with every metric.

Threshold rule (same for all methods, uses NO test labels):
    threshold = the `--quantile` quantile of the detector's scores on the
    training data.

Usage:
    python -m experiments.run_baseline --dataset synthetic
    python -m experiments.run_baseline --dataset smd --machine machine-1-1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import DEFAULT_DATA_DIR, load_smd, make_synthetic  # noqa: E402
from src.detectors import ZScoreDetector  # noqa: E402
from src.metrics import evaluate, segments  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def plot_scores(scores: np.ndarray, y_true: np.ndarray, threshold: float, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(scores, lw=0.7, color="#1f4e79", label="anomaly score")
    ax.axhline(threshold, color="#c0392b", lw=1, ls="--", label="threshold")
    for i, (s, e) in enumerate(segments(y_true)):
        ax.axvspan(s, e + 1, color="#f5b041", alpha=0.35, label="true anomaly" if i == 0 else None)
    ax.set_title(title)
    ax.set_xlabel("time step (test set)")
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smd", "synthetic"], default="synthetic")
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "smd":
        X_train, X_test, y_test = load_smd(args.machine, Path(args.data_dir))
        tag = f"smd_{args.machine}"
    else:
        X_train, X_test, y_test = make_synthetic(seed=args.seed)
        tag = f"synthetic_seed{args.seed}"

    print(
        f"{tag}: train {X_train.shape}, test {X_test.shape}, "
        f"anomaly rate {y_test.mean():.3%}, events {len(segments(y_test))}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for det in (ZScoreDetector("global"), ZScoreDetector("rolling", window=100)):
        det.fit(X_train)
        threshold = float(np.quantile(det.score(X_train), args.quantile))
        scores = det.score(X_test)
        metrics = evaluate(y_test, scores, threshold)
        rows.append({"dataset": tag, "method": det.name, **metrics})
        plot_scores(scores, y_test, threshold, f"{det.name} on {tag}", RESULTS_DIR / f"{tag}_{det.name}.png")

    table = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"baseline_{tag}.csv"
    table.to_csv(out_csv, index=False)
    show = ["method", "pw_f1", "pa_f1", "ev_f1", "ev_recall", "ev_precision", "auroc", "auprc"]
    print(table[show].round(3).to_string(index=False))
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
