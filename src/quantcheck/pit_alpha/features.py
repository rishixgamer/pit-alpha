"""Causal price transforms and fold-specific market volatility bands."""

from dataclasses import dataclass
from typing import cast

import polars as pl

from quantcheck.pit_alpha.alignment import DECISION_KEY
from quantcheck.pit_alpha.config import Metrics


def price_features(aligned: pl.DataFrame, price_column: str, lookback: int) -> pl.DataFrame:
    price = pl.col(price_column)
    previous = price.shift().over("symbol")
    previous_event = pl.col("event_ts").shift().over("symbol")
    return (
        aligned.sort("decision_ts", "symbol")
        .with_columns(
            pl.when((price > 0) & (previous > 0) & (pl.col("event_ts") > previous_event))
            .then(price / previous - 1)
            .otherwise(None)
            .alias("price_return")
        )
        .with_columns(
            pl.col("price_return")
            .rolling_std(window_size=lookback, min_samples=lookback, ddof=1)
            .over("symbol")
            .alias("trailing_volatility")
        )
    )


@dataclass(frozen=True)
class VolatilityBands:
    low: float | None
    high: float | None
    training_decisions: int

    def classify(self, market: pl.DataFrame) -> pl.DataFrame:
        volatility = pl.col("market_volatility")
        if self.low is None or self.high is None:
            return market.with_columns(pl.lit("unclassified").alias("regime"))
        return market.with_columns(
            pl.when(volatility.is_null())
            .then(pl.lit("unclassified"))
            .when(volatility <= self.low)
            .then(pl.lit("low"))
            .when(volatility <= self.high)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("high"))
            .alias("regime")
        )


def market_volatility(features: pl.DataFrame) -> pl.DataFrame:
    return (
        features.group_by("decision_ts")
        .agg(pl.col("trailing_volatility").median().alias("market_volatility"))
        .sort("decision_ts")
    )


def fit_bands(features: pl.DataFrame, training: pl.DataFrame, settings: Metrics) -> VolatilityBands:
    # Fit only from eligible symbol/decision rows. Classifying a live decision
    # can use the full market cross-section observable at that instant.
    series = market_volatility(
        features.join(training.select(*DECISION_KEY), on=DECISION_KEY, how="semi")
    )
    observed = series.get_column("market_volatility").drop_nulls()
    if len(observed) < 2:
        return VolatilityBands(None, None, len(observed))
    low = cast(float, observed.quantile(settings.low_quantile, interpolation="linear"))
    high = cast(float, observed.quantile(settings.high_quantile, interpolation="linear"))
    return VolatilityBands(low, high, len(observed))
