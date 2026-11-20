from dataclasses import replace
from datetime import timedelta
from importlib.resources import files

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantcheck.pit_alpha.alignment import UTC_TIMESTAMP
from quantcheck.pit_alpha.config import Candidate, RunConfig, Selection, load_config
from quantcheck.pit_alpha.example import START, synthetic_panel
from quantcheck.pit_alpha.metrics import Period, Statistic
from quantcheck.pit_alpha.models import fit
from quantcheck.pit_alpha.selection import check_sharpe
from quantcheck.pit_alpha.workflow import Run, run


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> RunConfig:
    path = tmp_path_factory.mktemp("pit-alpha") / "example.yaml"
    path.write_bytes(files("quantcheck.pit_alpha").joinpath("example.yaml").read_bytes())
    return load_config(path)


def with_missing_close_and_factor(frame: pl.DataFrame) -> pl.DataFrame:
    gap = (pl.col("symbol") == "S000") & (pl.col("event_ts") == START + timedelta(days=122))
    return frame.with_columns(
        pl.when(gap).then(None).otherwise(pl.col(name)).alias(name) for name in ("close", "factor")
    )


def assert_unchanged_before(left: Run, right: Run, cutoff_days: int) -> None:
    cutoff = pl.col("decision_ts") <= START + timedelta(days=cutoff_days)
    assert_frame_equal(
        left.features.filter(cutoff), right.features.filter(cutoff), check_exact=True
    )
    assert_frame_equal(
        left.predictions.filter(cutoff), right.predictions.filter(cutoff), check_exact=True
    )


def test_future_price_canary_crosses_fold_boundary_and_checks_masks(config: RunConfig) -> None:
    observations, labels = synthetic_panel()
    shifted_observations, shifted_labels = synthetic_panel(jump_index=152)
    base = run(config, with_missing_close_and_factor(observations), labels)
    shifted = run(config, with_missing_close_and_factor(shifted_observations), shifted_labels)
    assert base.selected.name == "ridge_factor"
    assert (
        base.predictions.filter(pl.col("decision_ts") <= START + timedelta(days=151))[
            "fold"
        ].n_unique()
        == 3
    )
    assert base.predictions["symbol"].n_unique() == 24
    assert base.predictions["prediction"].null_count() > 0
    assert_unchanged_before(base, shifted, 151)
    assert base.metrics.combined.annualized_return != shifted.metrics.combined.annualized_return
    changed_date = pl.col("decision_ts") == START + timedelta(days=121)
    changed_prediction = base.predictions.with_columns(
        pl.when(changed_date)
        .then(pl.col("prediction") + 1)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    with pytest.raises(AssertionError):
        assert_unchanged_before(base, replace(base, predictions=changed_prediction), 151)
    changed_feature = base.features.with_columns(
        pl.when(changed_date).then(None).otherwise(pl.col("price_return")).alias("price_return")
    )
    with pytest.raises(AssertionError):
        assert_unchanged_before(base, replace(base, features=changed_feature), 151)


def test_no_search_feedback_from_final_oos_and_all_trials_count(config: RunConfig) -> None:
    source, labels = synthetic_panel()
    control = run(config, source, labels)
    first_oos = START + timedelta(days=90)
    perturbed = labels.with_columns(
        pl.when(pl.col("decision_ts") >= first_oos)
        .then(-pl.col("target_return"))
        .otherwise(pl.col("target_return"))
        .alias("target_return")
    )
    shifted = run(config, source, perturbed)
    assert control.trials == shifted.trials
    assert control.selected == shifted.selected
    assert control.metrics.combined.sharpe != shifted.metrics.combined.sharpe
    assert control.folds[0].volatility_bands == shifted.folds[0].volatility_bands
    statuses = {trial.name: trial.status for trial in control.trials}
    assert statuses["duplicate_factor"] == "duplicate"
    assert statuses["missing_baseline"] == "failed"

    extra = config.candidates[0].model_copy(update={"name": "another_attempt"})
    extended = config.model_copy(update={"candidates": [*config.candidates, extra]})
    expanded = run(extended, source, labels)
    assert expanded.selected == control.selected
    assert expanded.sharpe_check.raw_p == control.sharpe_check.raw_p
    assert expanded.sharpe_check.attempted_trials == 6
    assert expanded.sharpe_check.adjusted_p == 0.003


def test_fold_local_fit_and_wrappers_distinguish_drop_and_median() -> None:
    times = [START + timedelta(days=i) for i in range(6)]
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * 6,
            "decision_ts": times,
            "x": [1.0, 3.0, float("nan"), 1000.0, None, None],
            "z": [1.0, None, 3.0, 1000.0, 1.5, None],
            "target_return": [0.01, 0.03, 0.02, 0.5, 0.2, 0.7],
        },
        schema_overrides={"decision_ts": UTC_TIMESTAMP},
    )
    train = panel.head(3)
    candidate = Candidate(
        name="ridge",
        wrapper="ridge",
        features=["x", "z"],
        missing="train_median",
        ridge_penalty=1.0,
    )
    model = fit(candidate, train)
    assert model.evidence.imputation == {"x": 2.0, "z": 2.0}
    assert model.evidence.centers == {"x": 2.0, "z": 2.0}
    assert model.evidence.missing_counts == {"x": 1, "z": 1}
    scores = model.predict(panel.tail(2))["prediction"].to_list()
    assert scores[0] is not None and scores[1] is None
    baseline = Candidate(name="baseline", wrapper="baseline", features=["x"], missing="drop")
    baseline_model = fit(baseline, train)
    assert baseline_model.evidence.usable_rows == 2
    assert baseline_model.predict(panel)["prediction"].to_list() == [
        1.0,
        3.0,
        None,
        1000.0,
        None,
        None,
    ]


def test_validation_changes_do_not_change_first_fold_fits(config: RunConfig) -> None:
    source, labels = synthetic_panel()
    base = run(config, source, labels)
    modified = source.with_columns(
        pl.when(pl.col("event_ts") >= START + timedelta(days=60))
        .then(pl.col("factor") + 1000)
        .otherwise(pl.col("factor"))
        .alias("factor")
    )
    changed = run(config, modified, labels)
    assert [trial.fit for trial in base.trials] == [trial.fit for trial in changed.trials]


def periods(values: list[float]) -> tuple[Period, ...]:
    missing = Statistic(None, 0)
    return tuple(
        Period(
            (START + timedelta(days=index)).isoformat(),
            1,
            "mid",
            4,
            4,
            4,
            missing,
            missing,
            0.0,
            value,
            "evaluated",
        )
        for index, value in enumerate(values)
    )


def test_bootstrap_on_seeded_serial_null_and_planted_signal() -> None:
    rng = np.random.default_rng(50)
    noise = rng.normal(scale=0.01, size=360)
    null = np.zeros(len(noise))
    for index in range(1, len(noise)):
        null[index] = 0.6 * null[index - 1] + noise[index]
    assert np.corrcoef(null[1:], null[:-1])[0, 1] > 0.5
    planted = null + 0.009
    settings = Selection(seed=92, block_length=9, bootstrap_samples=3999)
    for series, expected in ((null, False), (planted, True)):
        raw = float(np.sqrt(252) * series.mean() / series.std(ddof=1))
        result = check_sharpe(periods(series.tolist()), raw, 5, settings)
        assert result.passes is expected
        assert result.adjusted_p == min(1.0, 5 * (result.raw_p or 0.0))
        assert result == check_sharpe(periods(series.tolist()), raw, 5, settings)
    short = check_sharpe(periods([0.001] * 4), 0.0, 5, settings)
    assert short.reason == "insufficient_holdout_periods"
    constant = check_sharpe(periods([0.001] * 50), None, 5, settings)
    assert constant.reason == "undefined_holdout_sharpe"


def test_gapped_holdout_does_not_bootstrap_omitted_periods(config: RunConfig) -> None:
    source, labels = synthetic_panel()
    windows = config.windows.model_copy(update={"step_seconds": 31 * 86400})
    result = run(config.model_copy(update={"windows": windows}), source, labels)
    assert result.sharpe_check.omitted_decisions == 3
    assert result.sharpe_check.raw_sharpe is not None
    assert result.sharpe_check.adjusted_p is None
    assert result.sharpe_check.reason == "omitted_decisions_inside_holdout"


def test_overlapping_holdout_preserves_ic_and_withholds_portfolio_inference(
    config: RunConfig,
) -> None:
    source, labels = synthetic_panel()
    date = START + timedelta(days=110)
    overlapping = labels.with_columns(
        pl.when(pl.col("decision_ts") == date)
        .then(pl.col(name) + timedelta(days=1))
        .otherwise(pl.col(name))
        .alias(name)
        for name in ("label_end_ts", "label_available_ts")
    )
    result = run(config, source, overlapping)
    assert result.metrics.combined.ic.count == 150
    assert result.metrics.combined.annualized_return.reason == "overlapping_weighted_horizons"
    assert result.sharpe_check.reason == "overlapping_weighted_horizons"
    assert result.sharpe_check.passes is None
