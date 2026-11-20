"""Source-to-holdout factor workflow with a pre-holdout selection freeze."""

import hashlib
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl

from quantcheck import __version__
from quantcheck.pit_alpha.alignment import DECISION_KEY, asof_align, normalize_numbers
from quantcheck.pit_alpha.config import DERIVED_FEATURES, Metrics, RunConfig
from quantcheck.pit_alpha.features import (
    VolatilityBands,
    fit_bands,
    market_volatility,
    price_features,
)
from quantcheck.pit_alpha.metrics import Evaluation, evaluate_signals
from quantcheck.pit_alpha.models import FitEvidence, FittedModel, UnusableFit, fit
from quantcheck.pit_alpha.selection import SharpeCheck, check_sharpe
from quantcheck.pit_alpha.split import Eligibility, Fold, Window

REPORT_SCHEMA_VERSION = 1
SELECTION_RULE = "highest finite first-fold validation Sharpe; candidate order breaks ties"


@dataclass(frozen=True)
class Trial:
    name: str
    candidate_id: str
    configuration: dict[str, Any]
    status: str
    validation_sharpe: float | None = None
    validation_periods: int = 0
    reason: str | None = None
    duplicate_of: str | None = None
    fit: FitEvidence | None = None


@dataclass(frozen=True)
class FoldEvidence:
    number: int
    windows: dict[str, dict[str, str]]
    train: Eligibility
    validation: Eligibility
    oos_rows: int
    oos_unmatched: int
    oos_missing_features: dict[str, int]
    oos_missing_predictions: int
    model: FitEvidence
    volatility_bands: VolatilityBands


@dataclass(frozen=True)
class Run:
    configuration: dict[str, Any]
    observations: int
    decisions: int
    aligned_rows: int
    unmatched_rows: int
    trials: tuple[Trial, ...]
    selected: Trial
    folds: tuple[FoldEvidence, ...]
    metrics: Evaluation
    sharpe_check: SharpeCheck
    features: pl.DataFrame
    predictions: pl.DataFrame

    def report(self, input_hashes: dict[str, str]) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "quantcheck_version": __version__,
            "configuration": self.configuration,
            "input_sha256": input_hashes,
            "availability": {
                "timestamps": "UTC microseconds",
                "observation_cutoff": "event_ts <= available_ts <= decision_ts",
                "revision_order": "latest event, then latest availability",
                "missing_policy": "null/NaN are missing; no implicit forward fill",
                "observations": self.observations,
                "decisions": self.decisions,
                "aligned": self.aligned_rows,
                "unmatched": self.unmatched_rows,
            },
            "fold_policy": {
                "windows": "half-open [start, end); disjoint OOS windows",
                "embargo": "decision_ts < cutoff - embargo",
                "purge": "label_end_ts < cutoff",
                "label_cutoff": "label_available_ts <= cutoff",
                "fit_cutoff": "validation start",
                "selection_cutoff": "first OOS start",
            },
            "folds": [asdict(fold) for fold in self.folds],
            "selection": {
                "rule": SELECTION_RULE,
                "selected_name": self.selected.name,
                "selected_id": self.selected.candidate_id,
                "selection_fold": self.folds[0].number,
                "selection_windows": self.folds[0].windows,
                "attempted": len(self.trials),
                "failed": sum(trial.status == "failed" for trial in self.trials),
                "duplicates": sum(trial.status == "duplicate" for trial in self.trials),
                "trials": [asdict(trial) for trial in self.trials],
                "check": asdict(self.sharpe_check),
            },
            "metrics": asdict(self.metrics),
        }


def read_inputs(config: RunConfig, base: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]:
    obs_bytes = (base / config.inputs.observations).read_bytes()
    label_bytes = (base / config.inputs.decisions).read_bytes()
    return (
        pl.read_parquet(BytesIO(obs_bytes)),
        pl.read_parquet(BytesIO(label_bytes)),
        {
            "observations": hashlib.sha256(obs_bytes).hexdigest(),
            "decisions": hashlib.sha256(label_bytes).hexdigest(),
        },
    )


def _evaluate(predictions: pl.DataFrame, labels: pl.DataFrame, settings: Metrics) -> Evaluation:
    return evaluate_signals(
        predictions, labels, periods_per_year=settings.periods_per_year, cost_bps=settings.cost_bps
    )


def _score(
    model: FittedModel, rows: pl.DataFrame, number: int, regimes: pl.DataFrame | None = None
) -> pl.DataFrame:
    predictions = model.predict(rows).with_columns(pl.lit(number).alias("fold"))
    if regimes is not None:
        predictions = predictions.join(
            regimes.select("decision_ts", "regime"), on="decision_ts", how="left"
        )
    return predictions


def _training_rows(features: pl.DataFrame, fold: Fold) -> pl.DataFrame:
    return (
        fold.train.select(*DECISION_KEY, "target_return")
        .join(features, on=DECISION_KEY, how="left")
        .sort("decision_ts", "symbol")
    )


def _select(
    config: RunConfig, features: pl.DataFrame, fold: Fold
) -> tuple[tuple[Trial, ...], Trial, FittedModel]:
    trials: list[Trial] = []
    seen: dict[str, str] = {}
    winner: Trial | None = None
    selected: FittedModel | None = None

    # Outcomes can restrict which periods are eligible for selection. Within
    # those periods, position weights see every contemporaneous prediction.
    training = _training_rows(features, fold)
    dates = fold.validation.select("decision_ts").unique()
    validation_features = features.join(dates, on="decision_ts", how="semi")
    for candidate in config.candidates:
        design = candidate.model_dump(mode="json", exclude={"name"})
        identifier = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
        if identifier in seen:
            trials.append(
                Trial(
                    candidate.name, identifier, design, "duplicate", duplicate_of=seen[identifier]
                )
            )
            continue
        seen[identifier] = candidate.name
        model = None
        try:
            model = fit(candidate, training)
            scores = _score(model, validation_features, fold.number)
        except UnusableFit as error:
            trials.append(
                Trial(
                    candidate.name,
                    identifier,
                    design,
                    "failed",
                    reason=str(error),
                    fit=model.evidence if model else None,
                )
            )
            continue
        statistic = _evaluate(scores, fold.validation, config.metrics).combined.sharpe
        value = statistic.value
        if value is None:
            trials.append(
                Trial(
                    candidate.name,
                    identifier,
                    design,
                    "failed",
                    validation_periods=statistic.count,
                    reason=statistic.reason,
                    fit=model.evidence,
                )
            )
            continue
        trial = Trial(
            candidate.name,
            identifier,
            design,
            "evaluated",
            value,
            statistic.count,
            fit=model.evidence,
        )
        trials.append(trial)
        if winner is None or (
            winner.validation_sharpe is not None and value > winner.validation_sharpe
        ):
            winner, selected = trial, model

    if winner is None or selected is None:
        failures = "; ".join(f"{trial.name}: {trial.reason}" for trial in trials if trial.reason)
        raise ValueError(f"no candidate had an eligible validation Sharpe ({failures})")
    return tuple(trials), winner, selected


def _window(window: Window) -> dict[str, str]:
    return {"start": window.start.isoformat(), "end_exclusive": window.end.isoformat()}


def run(config: RunConfig, observations: pl.DataFrame, decisions: pl.DataFrame) -> Run:
    folds = tuple(config.windows.splitter().split(decisions))
    aligned = asof_align(observations, decisions, value_columns=config.inputs.observation_columns)
    features = normalize_numbers(
        price_features(aligned, config.inputs.price_column, config.metrics.volatility_lookback),
        DERIVED_FEATURES,
    )
    market = market_volatility(features)
    trials, selected, first_model = _select(config, features, folds[0])

    predictions: list[pl.DataFrame] = []
    manifests: list[FoldEvidence] = []
    for fold in folds:
        try:
            model = (
                first_model
                if fold.number == 1
                else fit(first_model.candidate, _training_rows(features, fold))
            )
        except UnusableFit as error:
            raise ValueError(
                f"selected candidate cannot fit fold {fold.number}: {error}"
            ) from error
        bands = fit_bands(features, fold.train, config.metrics)
        oos = features.filter(fold.oos_window.contains())
        scores = _score(
            model, oos, fold.number, bands.classify(market.filter(fold.oos_window.contains()))
        )
        predictions.append(scores)
        manifests.append(
            FoldEvidence(
                fold.number,
                {
                    "train": _window(fold.train_window),
                    "validation": _window(fold.validation_window),
                    "oos": _window(fold.oos_window),
                },
                fold.train_eligibility,
                fold.validation_eligibility,
                oos.height,
                oos.get_column("event_ts").null_count(),
                {name: oos.get_column(name).null_count() for name in model.candidate.features},
                scores.get_column("prediction").null_count(),
                model.evidence,
                bands,
            )
        )
    predicted = pl.concat(predictions).sort("decision_ts", "symbol")
    holdout = pl.concat([fold.oos for fold in folds]).with_columns(
        pl.when(pl.col("label_available_ts").is_not_null())
        .then(pl.col("target_return"))
        .otherwise(None)
        .alias("target_return")
    )
    evaluation = _evaluate(predicted, holdout, config.metrics)
    holdout_calendar = (
        features.filter(
            pl.col("decision_ts").is_between(
                folds[0].oos_window.start, folds[-1].oos_window.end, closed="left"
            )
        )
        .get_column("decision_ts")
        .n_unique()
    )
    check = check_sharpe(
        evaluation.periods,
        evaluation.combined.sharpe.value,
        len(trials),
        config.selection,
        omitted_decisions=holdout_calendar - len(evaluation.periods),
    )
    unmatched = aligned.get_column("event_ts").null_count()
    return Run(
        config.model_dump(mode="json", exclude={"output_directory"}),
        observations.height,
        decisions.height,
        aligned.height - unmatched,
        unmatched,
        trials,
        selected,
        tuple(manifests),
        evaluation,
        check,
        features,
        predicted,
    )
