from datetime import UTC, datetime, timedelta
from math import sqrt

import polars as pl
import pytest

from quantcheck.pit_alpha import evaluate_signals
from quantcheck.pit_alpha.alignment import UTC_TIMESTAMP
from quantcheck.pit_alpha.features import VolatilityBands

START = datetime(2024, 1, 1, tzinfo=UTC)


def market() -> tuple[pl.DataFrame, pl.DataFrame]:
    keys = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"] * 3,
            "decision_ts": [START + timedelta(days=day) for day in range(3) for _ in range(4)],
        },
        schema_overrides={"decision_ts": UTC_TIMESTAMP},
    )
    signals = keys.with_columns(
        pl.Series("fold", [1] * 4 + [2] * 8),
        pl.Series("prediction", [1, 1, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1]),
        pl.Series("regime", ["low"] * 4 + ["high"] * 4 + ["mid"] * 4),
    )
    outcomes = keys.with_columns(
        pl.Series(
            "target_return", [-0.10, 0.10, 0.20, 0.30, -0.20, 0.0, 0.20, 0.60, 0.0, 0.0, 0.0, 0.0]
        )
    )
    return signals, outcomes


def test_cross_sections_ties_drawdown_and_chronological_turnover() -> None:
    predictions, realized = market()
    result = evaluate_signals(
        predictions.reverse(), realized.reverse(), periods_per_year=3, cost_bps=100
    )
    first, second, third = result.periods
    assert first.rank_ic.value == pytest.approx(4.5 / sqrt(4.5 * 5))
    assert first.ic.value == pytest.approx(0.425 / sqrt(2.75 * 0.0875))
    assert third.ic.reason == "constant_predictions_or_returns"
    assert result.combined.ic.count == 2
    assert result.combined.rank_ic.count == 2
    assert result.combined.icir.count == 2
    assert [p.turnover for p in result.periods] == [0.5, 1.0, 0.5]
    assert [p.net_return for p in result.periods] == pytest.approx([0.12, -0.26, -0.005])
    assert result.combined.annualized_return.value == pytest.approx(1.12 * 0.74 * 0.995 - 1)
    assert result.combined.max_drawdown.value == pytest.approx(1 - 0.74 * 0.995)
    assert result.combined.turnover.value == pytest.approx(2 / 3)
    assert result.by_fold[2].max_drawdown.value == pytest.approx(1 - 0.74 * 0.995)
    assert result.by_regime["high"].turnover.value == 1
    assert result.by_fold_regime[1]["low"].paired_observations == 4
    assert second.ic.value is not None and second.ic.value < 0


def test_missing_returns_cannot_change_weights_and_metrics_declare_gaps() -> None:
    predictions, outcomes = market()
    missing_a = outcomes.with_columns(
        pl.when((pl.col("symbol") == "A") & (pl.col("decision_ts") == START))
        .then(None)
        .otherwise(pl.col("target_return"))
        .alias("target_return")
    )
    result = evaluate_signals(predictions, missing_a)
    assert result.periods[0].return_status == "missing_weighted_outcome"
    assert result.periods[0].turnover == 0.5
    assert result.periods[0].paired_count == 3
    assert result.combined.sharpe.reason == "incomplete_period_return_series"
    all_missing = predictions.with_columns(pl.lit(float("nan")).alias("prediction"))
    result = evaluate_signals(all_missing, outcomes)
    assert result.combined.predictions == 0
    assert result.combined.paired_observations == 0
    assert result.combined.ic.reason == "no_defined_cross_sections"
    assert result.combined.annualized_return.count == 0


def test_regime_endpoints_and_no_duplicate_holdout_keys() -> None:
    market_vol = pl.DataFrame({"market_volatility": [None, 0.0, 1.0, 1.5, 2.0, 2.1]})
    classified = VolatilityBands(1.0, 2.0, 5).classify(market_vol)
    assert classified["regime"].to_list() == ["unclassified", "low", "low", "mid", "mid", "high"]
    predictions, outcomes = market()
    with pytest.raises(ValueError, match="duplicate keys"):
        evaluate_signals(pl.concat([predictions, predictions.head(1)]), outcomes)
    mixed = predictions.with_columns(
        pl.when(pl.col("symbol") == "A").then(9).otherwise(pl.col("fold")).alias("fold")
    )
    with pytest.raises(ValueError, match="single fold and regime"):
        evaluate_signals(mixed, outcomes)


def test_pearson_ic_is_stable_with_large_finite_scores() -> None:
    predictions, outcomes = market()
    result = evaluate_signals(
        predictions.head(3).with_columns(pl.Series("prediction", [-1.6e308, 1.4e308, 1.6e308])),
        outcomes.head(3),
    )
    assert result.combined.ic.value == pytest.approx(474 / sqrt(5784 * 42))
