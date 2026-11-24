"""Reproduce a weekly cross-sectional study using pinned Coin Metrics archives."""

import argparse
import csv
import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from io import StringIO
from math import isfinite
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import polars as pl

from quantcheck.pit_alpha.config import RunConfig
from quantcheck.pit_alpha.reporting import write_reports
from quantcheck.pit_alpha.workflow import Run, read_inputs, run

REVISION = "f1a36afb962731c387bb03982758ab0103063da5"
ASSETS = ("ada", "bch", "btc", "doge", "eth", "ltc", "xlm", "xmr", "xrp", "zec")
FIRST = datetime(2020, 1, 7, tzinfo=UTC)
LAST = datetime(2025, 12, 30, tzinfo=UTC)
DAY = timedelta(days=1)
WEEK = timedelta(days=7)
COSTS = (0.0, 25.0, 100.0)
PRIMARY_COST = 25.0
SIGNALS = {"momentum_28d": "momentum", "reversal_28d": "reversal", "momentum_7d": "short_momentum"}


def weeks() -> Iterator[datetime]:
    date = FIRST
    while date <= LAST:
        yield date
        date += WEEK


def read_prices(content: bytes) -> dict[datetime, float]:
    prices = {}
    rows = csv.DictReader(StringIO(content.decode("utf-8")))
    if not {"time", "PriceUSD"} <= set(rows.fieldnames or ()):
        raise ValueError("missing time or PriceUSD column")
    for row in rows:
        if not row["PriceUSD"]:
            continue
        date = datetime.strptime(row["time"], "%Y-%m-%d").replace(tzinfo=UTC) + DAY
        price = float(row["PriceUSD"])
        if date in prices or not isfinite(price) or price <= 0:
            raise ValueError(f"invalid or duplicate reference rate on {date.date()}")
        prices[date] = price
    return prices


def panels(
    prices_by_asset: dict[str, dict[datetime, float]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    observations, labels = [], []
    for asset, prices in prices_by_asset.items():
        for decision in weeks():
            event = decision - 2 * DAY
            end = decision + WEEK
            try:
                close = prices[event]
                momentum = close / prices[event - 4 * WEEK] - 1
                short_momentum = close / prices[event - WEEK] - 1
                return_ = prices[end] / prices[decision] - 1
            except KeyError as error:
                raise ValueError(f"{asset}: missing rate at {error.args[0]}") from error
            observations.append(
                {
                    "symbol": asset,
                    "event_ts": event,
                    "available_ts": decision,
                    "close": close,
                    "momentum": momentum,
                    "reversal": -momentum,
                    "short_momentum": short_momentum,
                }
            )
            labels.append(
                {
                    "symbol": asset,
                    "decision_ts": decision,
                    "label_end_ts": end,
                    "label_available_ts": end + 2 * DAY,
                    "target_return": return_,
                }
            )
    return pl.DataFrame(observations), pl.DataFrame(labels)


def settings(signals: tuple[str, ...], cost: float, directory: str) -> RunConfig:
    week = 604800
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "inputs": {
                "observations": "observations.parquet",
                "decisions": "decisions.parquet",
                "price_column": "close",
                "observation_columns": ["close", *SIGNALS.values()],
            },
            "windows": {
                "train_seconds": 104 * week,
                "validation_seconds": 26 * week,
                "oos_seconds": 26 * week,
                "step_seconds": 26 * week,
                "embargo_seconds": 86400,
                "mode": "rolling",
            },
            "candidates": [
                {
                    "name": name,
                    "wrapper": "baseline",
                    "features": [SIGNALS[name]],
                    "missing": "drop",
                }
                for name in signals
            ],
            "metrics": {"periods_per_year": 52, "cost_bps": cost},
            "selection": {"block_length": 8, "min_holdout_periods": 52, "seed": 1931},
            "output_directory": directory,
        }
    )


def weekly_mean_interval(result: Run, config: RunConfig) -> list[float]:
    """Percentile interval for the selected rule's mean weekly net return."""
    observed = [period.net_return for period in result.metrics.periods]
    if any(value is None for value in observed):
        raise ValueError("cannot bootstrap an incomplete holdout")
    returns = np.asarray(observed, dtype=np.float64)
    count = len(returns)
    length = config.selection.block_length
    rng = np.random.default_rng(config.selection.seed)
    starts = rng.integers(
        0, count, size=(config.selection.bootstrap_samples, (count + length - 1) // length)
    )
    indices = ((starts[:, :, None] + np.arange(length)) % count).reshape(len(starts), -1)
    means = returns[indices[:, :count]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    pinned = json.loads(Path(__file__).with_name("SOURCES.json").read_text())
    sources = {}
    prices = {}
    for asset in ASSETS:
        url = f"https://raw.githubusercontent.com/coinmetrics/data/{REVISION}/csv/{asset}.csv"
        with urlopen(url, timeout=30) as response:
            content = response.read()
        sources[asset] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
        if sources[asset] != pinned[asset]:
            raise ValueError(f"{asset}: source identity differs from the pinned manifest")
        prices[asset] = read_prices(content)
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    observations, labels = panels(prices)
    observations.write_parquet(root / "observations.parquet")
    labels.write_parquet(root / "decisions.parquet")

    primary = settings(tuple(SIGNALS), PRIMARY_COST, "primary")
    inputs = read_inputs(primary, root)
    result = run(primary, *inputs[:2])
    write_reports(root / primary.output_directory, result, inputs[2])
    interval = weekly_mean_interval(result, primary)
    (root / "weekly_mean_interval.json").write_text(
        json.dumps({"percentile_95_weekly_mean_return": interval}, indent=2) + "\n"
    )
    for cost in COSTS:
        for name in SIGNALS:
            config = settings((name,), cost, f"{name}-{cost:g}bps")
            write_reports(root / config.output_directory, run(config, *inputs[:2]), inputs[2])


if __name__ == "__main__":
    main()
