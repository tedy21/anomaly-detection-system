# Anomaly Detection in Time Series: Drift, Few Labels, and Metric Choice

A small, reproducible study comparing anomaly detectors on server-sensor data
(Server Machine Dataset) and, later, payment data. Status: **Week 1-2 done**
(data loader, metrics, z-score baselines). Isolation Forest, the autoencoder,
and the experiments are added in later weeks.

This repository is the GitHub project for the anomaly-detection-system study.

## Research questions

1. How does each method behave when normal behaviour slowly drifts?
2. How does each method behave when only a few anomalies are labelled?
3. Does the choice of metric (point-wise, point-adjusted, event-based) change
   the ranking of methods?

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

```bash
python -m src.data --download --machines machine-1-1   # needs internet
python -m pytest -q                                    # unit tests
python -m experiments.run_baseline --dataset smd --machine machine-1-1
```

Results (CSV and plots) are written to `results/`. To check that the pipeline
runs without any download, use `--dataset synthetic`. Synthetic data is only a
smoke test, never a result to report.

## Evaluation rules

- Data is split by time; nothing is shuffled.
- The threshold is the 99.5th percentile of the detector's scores on the
  training data. No test labels are used to choose it.
- Metrics: point-wise, point-adjusted and event-based precision/recall/F1,
  plus AUROC and AUPRC (threshold-free).
- Point-adjusted F1 is known to overstate performance; it is reported only to
  measure how much it changes the ranking (question 3).

## Layout

```
src/data.py          SMD download/loader, synthetic generator
src/metrics.py       point-wise, point-adjusted, event-based, AUROC/AUPRC
src/detectors.py     z-score baselines (global, rolling)
experiments/         one script per experiment
tests/               unit tests for metrics and detectors
results/             generated tables and plots (not hand-edited)
```

## Known limitations

- The rolling z-score can be fooled by long anomalies: the window absorbs the
  anomaly, so later points of the same event score lower.
- Score = maximum |z| over features, which is noisy when many features are
  near-constant. A small std floor reduces this but does not remove it.
- Dataset: SMD only so far. Check the dataset licence at its source page
  before redistributing any data.

## Roadmap

| Week | Task | Status |
| --- | --- | --- |
| 1-2 | Loader, metrics, z-score baseline | done |
| 3 | Isolation Forest | todo |
| 4 | Autoencoder (PyTorch) | todo |
| 5 | Drift and few-label experiments | todo |
| 6 | Metric-choice experiment, optional PaySim | todo |
| 7-8 | Report and cleanup | todo |

