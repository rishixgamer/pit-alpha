"""Candidate accounting and the one-sided post-selection Sharpe check."""

from dataclasses import dataclass

import numpy as np

from quantcheck.pit_alpha.config import Selection
from quantcheck.pit_alpha.metrics import Period


@dataclass(frozen=True)
class SharpeCheck:
    method: str
    status: str
    reason: str | None
    holdout_periods: int
    observed_periods: int
    attempted_trials: int
    raw_sharpe: float | None
    raw_p: float | None
    adjusted_p: float | None
    passes: bool | None
    minimum_adjusted_p: float
    omitted_decisions: int


def check_sharpe(
    periods: tuple[Period, ...],
    raw_sharpe: float | None,
    trials: int,
    settings: Selection,
    *,
    omitted_decisions: int = 0,
) -> SharpeCheck:
    """Test positive Sharpe with centered circular blocks and Bonferroni adjustment."""
    returns = np.asarray(
        [period.net_return for period in periods if period.net_return is not None], dtype=np.float64
    )
    method = "centered_circular_block_bootstrap_bonferroni"
    minimum_p = min(1.0, trials / (1 + settings.bootstrap_samples))
    reason = None
    if omitted_decisions:
        reason = "omitted_decisions_inside_holdout"
    elif any(period.return_status == "overlapping_weighted_horizon" for period in periods):
        reason = "overlapping_weighted_horizons"
    elif len(returns) != len(periods):
        reason = "incomplete_holdout_return_series"
    elif len(returns) < max(settings.min_holdout_periods, 4 * settings.block_length):
        reason = "insufficient_holdout_periods"
    elif raw_sharpe is None:
        reason = "undefined_holdout_sharpe"
    if reason is not None:
        return SharpeCheck(
            method,
            "undefined",
            reason,
            len(periods),
            len(returns),
            trials,
            raw_sharpe,
            None,
            None,
            None,
            minimum_p,
            omitted_decisions,
        )

    rng = np.random.default_rng(settings.seed)
    normalized = returns / np.abs(returns).max()
    observed = float(normalized.mean() / normalized.std(ddof=1))
    centered = normalized - normalized.mean()
    count = len(centered)
    length = settings.block_length
    blocks = (count + length - 1) // length
    offsets = np.arange(length)
    extreme = 0
    batch_size = max(1, min(256, 262_144 // count))
    for start in range(0, settings.bootstrap_samples, batch_size):
        size = min(batch_size, settings.bootstrap_samples - start)
        starts = rng.integers(0, count, size=(size, blocks))
        indices = (starts[:, :, None] + offsets) % count
        sample = centered[indices.reshape(size, -1)[:, :count]]
        means = sample.mean(axis=1)
        deviations = sample.std(axis=1, ddof=1)
        # A degenerate resample has the limiting signed Sharpe, or zero if its mean is zero.
        scores = np.select([means > 0, means < 0], [np.inf, -np.inf], default=0.0)
        np.divide(means, deviations, out=scores, where=deviations > 0)
        extreme += int(np.count_nonzero(scores >= observed))
    raw_p = (1 + extreme) / (1 + settings.bootstrap_samples)
    adjusted = min(1.0, raw_p * trials)
    return SharpeCheck(
        method,
        "evaluated",
        None,
        len(periods),
        count,
        trials,
        raw_sharpe,
        raw_p,
        adjusted,
        adjusted <= settings.alpha,
        minimum_p,
        omitted_decisions,
    )
