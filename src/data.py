"""Data loading.

* Server Machine Dataset (SMD): 28 servers, 38 sensor features each.
  Source: https://github.com/NetManAIOps/OmniAnomaly (check the licence there
  before redistributing; this repo only downloads, never stores the data).
* make_synthetic(): a small generated dataset used for tests and quick checks.

Download SMD (needs internet access to raw.githubusercontent.com):

    python -m src.data --download --machines machine-1-1 machine-2-1
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

SMD_BASE = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
    "master/ServerMachineDataset"
)
SMD_SPLITS = ("train", "test", "test_label")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smd"


def smd_machine_names() -> list[str]:
    """machine-1-1 ... machine-1-8, machine-2-1 ... 2-9, machine-3-1 ... 3-11."""
    counts = {1: 8, 2: 9, 3: 11}
    return [f"machine-{g}-{i}" for g, n in counts.items() for i in range(1, n + 1)]


def download_smd(machines: list[str], data_dir: Path = DEFAULT_DATA_DIR) -> None:
    for machine in machines:
        for split in SMD_SPLITS:
            target = Path(data_dir) / split / f"{machine}.txt"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{SMD_BASE}/{split}/{machine}.txt"
            print(f"downloading {url}")
            try:
                urllib.request.urlretrieve(url, target)
            except Exception as exc:  # noqa: BLE001
                if target.exists():
                    target.unlink()
                raise RuntimeError(
                    f"Could not download {url}: {exc}. "
                    "Download the file manually and place it at "
                    f"{target}"
                ) from exc


def load_smd(machine: str = "machine-1-1", data_dir: Path = DEFAULT_DATA_DIR):
    """Return X_train (n, 38), X_test (m, 38), y_test (m,) for one machine."""
    data_dir = Path(data_dir)
    paths = {s: data_dir / s / f"{machine}.txt" for s in SMD_SPLITS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "SMD files not found. Run: python -m src.data --download "
            f"--machines {machine}\nMissing: {missing}"
        )
    X_train = np.loadtxt(paths["train"], delimiter=",")
    X_test = np.loadtxt(paths["test"], delimiter=",")
    y_test = np.loadtxt(paths["test_label"]).astype(int)
    return X_train, X_test, y_test


def make_synthetic(
    seed: int = 0,
    n_train: int = 3000,
    n_test: int = 3000,
    n_features: int = 8,
    n_events: int = 6,
    event_length: tuple[int, int] = (10, 40),
):
    """Daily-cycle sensors + noise. Test data gets level-shift / spike events.

    Only for tests and pipeline checks. Do NOT report results on it as if it
    were a real benchmark.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    t = np.arange(n)
    phases = rng.uniform(0, 2 * np.pi, n_features)
    periods = rng.uniform(100, 300, n_features)
    base = np.sin(2 * np.pi * t[:, None] / periods + phases)
    X = base + 0.1 * rng.standard_normal((n, n_features))

    y = np.zeros(n, dtype=int)
    starts = np.sort(rng.choice(np.arange(n_train + 50, n - 60), n_events, replace=False))
    last_end = -1
    for s in starts:
        if s <= last_end:
            continue
        length = int(rng.integers(*event_length))
        cols = rng.choice(n_features, size=max(1, n_features // 3), replace=False)
        X[s : s + length, cols] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0)
        y[s : s + length] = 1
        last_end = s + length
    return X[:n_train], X[n_train:], y[n_train:]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--machines", nargs="+", default=["machine-1-1"])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    if args.download:
        download_smd(args.machines, Path(args.data_dir))
        print("done")


if __name__ == "__main__":
    _main()
