"""Deterministic baseline and ridge wrappers fit on eligible training rows only."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quantcheck.pit_alpha.alignment import DECISION_KEY
from quantcheck.pit_alpha.config import Candidate

FloatArray = NDArray[np.float64]


class UnusableFit(ValueError):
    """The wrapper could not fit the candidate on its eligible training rows."""


@dataclass(frozen=True)
class FitEvidence:
    training_rows: int
    usable_rows: int
    missing_counts: dict[str, int]
    imputation: dict[str, float]
    centers: dict[str, float]
    scales: dict[str, float]
    coefficients: dict[str, float]
    intercept: float


@dataclass(frozen=True)
class FittedModel:
    candidate: Candidate
    evidence: FitEvidence

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        columns = self.candidate.features
        raw = _matrix(features, columns)
        values, keep = _prepare(raw, self.candidate, self.evidence.imputation)
        scores = np.full(len(raw), np.nan)
        if self.candidate.wrapper == "baseline":
            scores[keep] = values[:, 0]
        else:
            centers = np.array([self.evidence.centers[name] for name in columns])
            scales = np.array([self.evidence.scales[name] for name in columns])
            coefficients = np.array([self.evidence.coefficients[name] for name in columns])
            scores[keep] = ((values - centers) / scales) @ coefficients + self.evidence.intercept
        if not np.isfinite(scores[keep]).all():
            raise UnusableFit(f"candidate {self.candidate.name} produced a nonfinite prediction")
        return features.select(*DECISION_KEY).with_columns(
            pl.Series("prediction", scores).fill_nan(None)
        )


def _matrix(frame: pl.DataFrame, columns: list[str]) -> FloatArray:
    return np.asarray(frame.select(columns).to_numpy(), dtype=np.float64)


def _prepare(
    values: FloatArray, candidate: Candidate, medians: dict[str, float]
) -> tuple[FloatArray, NDArray[np.bool_]]:
    missing = np.isnan(values)
    if candidate.missing == "drop":
        keep = ~missing.any(axis=1)
        return values[keep], keep
    keep = ~missing.all(axis=1)
    replacements = np.array([medians[name] for name in candidate.features])
    return np.where(missing[keep], replacements, values[keep]), keep


def fit(candidate: Candidate, rows: pl.DataFrame) -> FittedModel:
    columns = candidate.features
    raw = _matrix(rows, columns)
    missing = np.isnan(raw)
    medians: dict[str, float] = {}
    if candidate.missing == "train_median":
        if missing.all(axis=0).any():
            raise UnusableFit("a feature has no observed eligible training values")
        medians = dict(zip(columns, np.nanmedian(raw, axis=0).tolist(), strict=True))
    values, keep = _prepare(raw, candidate, medians)
    min_rows = 2 if candidate.wrapper == "ridge" else 1
    if len(values) < min_rows:
        raise UnusableFit(f"fewer than {min_rows} usable training rows")
    missing_counts = dict(zip(columns, missing.sum(axis=0).tolist(), strict=True))
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    coefficients = dict.fromkeys(columns, 1.0)
    intercept = 0.0
    if candidate.wrapper == "ridge":
        penalty = candidate.ridge_penalty
        assert penalty is not None
        x_mean = values.mean(axis=0)
        x_scale = values.std(axis=0)
        x_scale[x_scale == 0] = 1.0
        x = (values - x_mean) / x_scale
        y = rows.get_column("target_return").to_numpy()[keep]
        intercept = float(y.mean())
        system = x.T @ x + penalty * np.eye(len(columns))
        try:
            coefs = np.linalg.solve(system, x.T @ (y - intercept))
        except np.linalg.LinAlgError as error:
            raise UnusableFit("ridge system could not be solved") from error
        if not (np.isfinite(coefs).all() and np.isfinite(x).all() and np.isfinite(intercept)):
            raise UnusableFit("ridge fit has nonfinite parameters")
        centers = dict(zip(columns, x_mean.tolist(), strict=True))
        scales = dict(zip(columns, x_scale.tolist(), strict=True))
        coefficients = dict(zip(columns, coefs.tolist(), strict=True))

    return FittedModel(
        candidate,
        FitEvidence(
            rows.height,
            len(values),
            missing_counts,
            medians,
            centers,
            scales,
            coefficients,
            intercept,
        ),
    )
