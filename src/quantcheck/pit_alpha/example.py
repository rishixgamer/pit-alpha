"""Seeded synthetic panel used by the packaged example and leakage canary."""

from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from quantcheck.pit_alpha.alignment import UTC_TIMESTAMP

EXAMPLE_SEED = 712
EXAMPLE_DATES = 241
EXAMPLE_SYMBOLS = 24
START = datetime(2024, 1, 1, 16, tzinfo=UTC)


def synthetic_panel(
    *,
    dates: int = EXAMPLE_DATES,
    symbols: int = EXAMPLE_SYMBOLS,
    seed: int = EXAMPLE_SEED,
    jump_index: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    shape = (dates + 1, symbols)
    signal = rng.normal(size=shape)
    noise = rng.normal(size=(dates, symbols))
    multiplier = np.where(np.arange(dates) % 45 < 20, 0.006, 0.020)
    log_changes = 0.010 * signal[:-1] + noise * multiplier[:, None]
    prices = np.vstack((np.zeros(symbols), np.cumsum(log_changes, axis=0)))
    prices = 100 * np.exp(prices)
    if jump_index is not None:
        prices[jump_index, ::2] *= 1.5
    outcomes = prices[1:] / prices[:-1] - 1

    names = [f"S{index:03d}" for index in range(symbols)]
    timestamps = [START + timedelta(days=index) for index in range(dates + 1)]
    all_decisions = [timestamp for timestamp in timestamps[:-1] for _ in names]
    ends = [timestamp for timestamp in timestamps[1:] for _ in names]
    observations = pl.DataFrame(
        {
            "symbol": names * dates,
            "event_ts": all_decisions,
            "available_ts": all_decisions,
            "close": prices[:-1].ravel(),
            "factor": signal[:-1].ravel(),
            "unobserved": pl.Series([None] * (dates * symbols), dtype=pl.Float64),
        },
        schema_overrides={"event_ts": UTC_TIMESTAMP, "available_ts": UTC_TIMESTAMP},
    )
    targets = pl.DataFrame(
        {
            "symbol": names * dates,
            "decision_ts": all_decisions,
            "label_end_ts": ends,
            "label_available_ts": ends,
            "target_return": outcomes.ravel(),
        },
        schema_overrides={
            "decision_ts": UTC_TIMESTAMP,
            "label_end_ts": UTC_TIMESTAMP,
            "label_available_ts": UTC_TIMESTAMP,
        },
    )
    return observations, targets


def write_example(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / name for name in ("example.yaml", "observations.parquet", "decisions.parquet")
    ]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("example files already exist; choose an unused example directory")
    config, observations_path, decisions_path = paths
    observations, decisions = synthetic_panel()
    observations.write_parquet(observations_path)
    decisions.write_parquet(decisions_path)
    config.write_bytes(files("quantcheck.pit_alpha").joinpath("example.yaml").read_bytes())
    return config
