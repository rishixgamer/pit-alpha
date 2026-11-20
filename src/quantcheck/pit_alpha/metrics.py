"""Cross-sectional factor metrics and a chronological, hypothetical portfolio."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from math import expm1, fsum, isfinite, log1p, sqrt
from statistics import mean, stdev
from sys import float_info

import polars as pl

from quantcheck.pit_alpha.alignment import (
    DECISION_KEY,
    normalize_numbers,
    normalize_panel,
    require_unique,
)

REGIMES = ("low", "mid", "high", "unclassified")


@dataclass(frozen=True)
class Statistic:
    value: float | None
    count: int
    reason: str | None = None


@dataclass(frozen=True)
class Period:
    decision_ts: str
    fold: int
    regime: str
    symbol_count: int
    scored_count: int
    paired_count: int
    ic: Statistic
    rank_ic: Statistic
    turnover: float
    net_return: float | None
    return_status: str


@dataclass(frozen=True)
class Summary:
    decisions: int
    predictions: int
    paired_observations: int
    ic: Statistic
    rank_ic: Statistic
    icir: Statistic
    annualized_return: Statistic
    sharpe: Statistic
    max_drawdown: Statistic
    turnover: Statistic


@dataclass(frozen=True)
class Evaluation:
    combined: Summary
    by_fold: dict[int, Summary]
    by_regime: dict[str, Summary]
    by_fold_regime: dict[int, dict[str, Summary]]
    periods: tuple[Period, ...]


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    for _, members in groupby(ordered, key=values.__getitem__):
        indices = list(members)
        end = start + len(indices)
        for index in indices:
            ranks[index] = (start + 1 + end) / 2
        start = end
    return ranks


def _center(values: Sequence[float]) -> list[float]:
    # Halve large values to bound subtraction when the signs differ.
    if max(map(abs, values)) > float_info.max / 2:
        values = [value / 2 for value in values]
    average = mean(values)
    return [value - average for value in values]


def _correlation(left: Sequence[float], right: Sequence[float]) -> Statistic:
    count = len(left)
    if count < 2:
        return Statistic(None, count, "fewer_than_two_pairs")
    x, y = _center(left), _center(right)
    x_scale = max(map(abs, x))
    y_scale = max(map(abs, y))
    if x_scale == 0 or y_scale == 0:
        return Statistic(None, count, "constant_predictions_or_returns")
    x = [value / x_scale for value in x]
    y = [value / y_scale for value in y]
    denominator = sqrt(fsum(value * value for value in x) * fsum(value * value for value in y))
    coefficient = fsum(a * b for a, b in zip(x, y, strict=True)) / denominator
    return Statistic(min(1.0, max(-1.0, coefficient)), count)


def _weights(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    ordered = sorted(scores.values())
    lower, upper = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
    long = [symbol for symbol, value in scores.items() if value >= upper and value > lower]
    short = [symbol for symbol, value in scores.items() if value <= lower and value < upper]
    if not long or not short:
        return {}
    return {symbol: 0.5 / len(long) for symbol in long} | {
        symbol: -0.5 / len(short) for symbol in short
    }


def _periods(joined: pl.DataFrame, cost: float) -> tuple[Period, ...]:
    periods: list[Period] = []
    previous: dict[str, float] = {}
    for _, rows in groupby(joined.iter_rows(named=True), key=lambda row: row["decision_ts"]):
        members = list(rows)
        first = members[0]
        if any(row["fold"] != first["fold"] or row["regime"] != first["regime"] for row in members):
            raise ValueError("each decision_ts must belong to a single fold and regime")
        scores = {
            row["symbol"]: row["prediction"] for row in members if row["prediction"] is not None
        }
        outcomes = {
            row["symbol"]: row["target_return"]
            for row in members
            if row["target_return"] is not None
        }
        paired = [symbol for symbol in scores if symbol in outcomes]
        left = [scores[symbol] for symbol in paired]
        right = [outcomes[symbol] for symbol in paired]
        weights = _weights(scores)
        turnover = 0.5 * fsum(
            abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in sorted(weights.keys() | previous.keys())
        )
        net_return = None
        if not scores:
            status = "no_finite_predictions"
        elif weights.keys() - outcomes.keys():
            status = "missing_weighted_outcome"
        elif any(row.get("overlapping_horizon") and row["symbol"] in weights for row in members):
            status = "overlapping_weighted_horizon"
        else:
            net_return = fsum(weight * outcomes[symbol] for symbol, weight in weights.items())
            net_return -= turnover * cost
            status = "evaluated"
        periods.append(
            Period(
                first["decision_ts"].isoformat(),
                first["fold"],
                first["regime"],
                len(members),
                len(scores),
                len(paired),
                _correlation(left, right),
                _correlation(_average_ranks(left), _average_ranks(right)),
                turnover,
                net_return,
                status,
            )
        )
        previous = weights
    return tuple(periods)


def _average(values: Sequence[float], reason: str) -> Statistic:
    return Statistic(mean(values), len(values)) if values else Statistic(None, 0, reason)


def _ratio(values: Sequence[float], *, scale: float = 1.0) -> Statistic:
    if len(values) < 2:
        return Statistic(None, len(values), "fewer_than_two_periods")
    deviation = stdev(values)
    if deviation == 0:
        return Statistic(None, len(values), "zero_sample_deviation")
    return Statistic(scale * mean(values) / deviation, len(values))


def _equity(returns: Sequence[float], annualization: int) -> tuple[Statistic, Statistic]:
    count = len(returns)
    if not count:
        missing = Statistic(None, 0, "no_evaluable_returns")
        return missing, missing
    log_wealth = peak = worst = 0.0
    for value in returns:
        if value <= -1:
            # The hypothetical portfolio runs out of capital at this point.
            return Statistic(-1.0, count), Statistic(1.0, count)
        log_wealth += log1p(value)
        peak = max(peak, log_wealth)
        worst = max(worst, -expm1(log_wealth - peak))
    annual_log_return = log_wealth * annualization / count
    try:
        annual = Statistic(expm1(annual_log_return), count)
    except OverflowError:
        annual = Statistic(None, count, "annualized_return_overflow")
    return annual, Statistic(worst, count)


def _summarize(periods: Sequence[Period], annualization: int) -> Summary:
    ic = [period.ic.value for period in periods if period.ic.value is not None]
    rank = [period.rank_ic.value for period in periods if period.rank_ic.value is not None]
    observed = [period.net_return for period in periods if period.net_return is not None]
    # Gaps cannot be treated as zero excess returns or silently compounded away.
    # Costs following a missing period also depend on the preceding intended weights.
    if len(observed) < len(periods):
        reason = (
            "overlapping_weighted_horizons"
            if any(period.return_status == "overlapping_weighted_horizon" for period in periods)
            else "incomplete_period_return_series"
        )
        annual_return = sharpe = drawdown = Statistic(None, len(observed), reason)
    else:
        annual_return, drawdown = _equity(observed, annualization)
        sharpe = _ratio(observed, scale=sqrt(annualization))
    return Summary(
        len(periods),
        sum(period.scored_count for period in periods),
        sum(period.paired_count for period in periods),
        _average(ic, "no_defined_cross_sections"),
        _average(rank, "no_defined_cross_sections"),
        _ratio(ic),
        annual_return,
        sharpe,
        drawdown,
        _average([period.turnover for period in periods], "no_decisions"),
    )


def _regime_summaries(periods: Sequence[Period], annualization: int) -> dict[str, Summary]:
    return {
        regime: _summarize([period for period in periods if period.regime == regime], annualization)
        for regime in REGIMES
    }


def evaluate_signals(
    predictions: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    periods_per_year: int = 252,
    cost_bps: float = 0.0,
) -> Evaluation:
    """Evaluate unique keyed OOS predictions against simple, per-period realized returns.

    Predictions require symbol, decision_ts, fold and prediction. The optional
    regime column must identify one causal regime per decision instant. Outcome
    rows require symbol, decision_ts and target_return. An optional
    label_end_ts allows the portfolio to detect overlapping weighted horizons.
    """
    if type(periods_per_year) is not int or not 1 <= periods_per_year <= 10_000_000:
        raise ValueError("periods_per_year must be an integer between 1 and 10000000")
    if not isfinite(cost_bps) or not 0 <= cost_bps <= 10_000:
        raise ValueError("cost_bps must be finite and between 0 and 10000")
    signals = normalize_numbers(
        normalize_panel(predictions, timestamps=("decision_ts",)), ("prediction",)
    )
    require_unique(signals, DECISION_KEY, "OOS predictions")
    fold_dtype = signals.schema.get("fold")
    if (
        fold_dtype is None
        or not fold_dtype.is_integer()
        or signals.select((pl.col("fold") < 1).fill_null(True).any()).item()
    ):
        raise ValueError("fold must contain positive integer IDs")
    if "regime" not in signals.columns:
        signals = signals.with_columns(pl.lit("unclassified").alias("regime"))
    if (
        signals.schema["regime"] != pl.String
        or signals.select((~pl.col("regime").is_in(REGIMES)).fill_null(True).any()).item()
    ):
        raise ValueError("regime must be low, mid, high, or unclassified")

    horizon = ("label_end_ts",) if "label_end_ts" in outcomes.columns else ()
    realized = normalize_numbers(
        normalize_panel(outcomes, timestamps=("decision_ts", *horizon), nullable=horizon),
        ("target_return",),
    )
    if horizon and realized.select((pl.col("label_end_ts") <= pl.col("decision_ts")).any()).item():
        raise ValueError("label_end_ts must be after decision_ts")
    require_unique(realized, DECISION_KEY, "outcomes")
    if realized.select((pl.col("target_return") < -1).any()).item():
        raise ValueError("simple asset returns cannot be below -1")
    joined = signals.select(*DECISION_KEY, "prediction", "fold", "regime").join(
        realized.select(*DECISION_KEY, "target_return", *horizon), on=DECISION_KEY, how="left"
    )
    if horizon:
        timeline = (
            signals.select("decision_ts")
            .unique()
            .sort("decision_ts")
            .with_columns(pl.col("decision_ts").shift(-1).alias("next_decision_ts"))
        )
        joined = joined.join(timeline, on="decision_ts").with_columns(
            (pl.col("label_end_ts") > pl.col("next_decision_ts")).alias("overlapping_horizon")
        )
    periods = _periods(joined.sort("decision_ts", "symbol"), cost_bps / 10_000)
    groups: dict[int, list[Period]] = {}
    for period in periods:
        groups.setdefault(period.fold, []).append(period)
    return Evaluation(
        _summarize(periods, periods_per_year),
        {fold: _summarize(rows, periods_per_year) for fold, rows in groups.items()},
        _regime_summaries(periods, periods_per_year),
        {fold: _regime_summaries(rows, periods_per_year) for fold, rows in groups.items()},
        periods,
    )
