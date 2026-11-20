"""Versioned YAML configuration and predeclared research choices."""

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantcheck.pit_alpha.split import PurgedWalkForwardSplit

DERIVED_FEATURES = ("price_return", "trailing_volatility")
RESERVED_COLUMNS = {
    "symbol",
    "event_ts",
    "available_ts",
    "decision_ts",
    "label_end_ts",
    "label_available_ts",
    "target_return",
    "prediction",
    "fold",
    "regime",
    *DERIVED_FEATURES,
}
Name = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Seconds = Annotated[int, Field(gt=0, le=315_576_000)]
Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Inputs(_Config):
    observations: Annotated[str, Field(min_length=1)]
    decisions: Annotated[str, Field(min_length=1)]
    observation_columns: Annotated[list[Name], Field(min_length=1)]
    price_column: Name


class Windows(_Config):
    train_seconds: Seconds
    validation_seconds: Seconds
    oos_seconds: Seconds
    step_seconds: Seconds
    embargo_seconds: Annotated[int, Field(ge=0, le=315_576_000)] = 0
    mode: Literal["rolling", "expanding"] = "rolling"

    def splitter(self) -> PurgedWalkForwardSplit:
        return PurgedWalkForwardSplit(
            train=timedelta(seconds=self.train_seconds),
            validation=timedelta(seconds=self.validation_seconds),
            oos=timedelta(seconds=self.oos_seconds),
            step=timedelta(seconds=self.step_seconds),
            embargo=timedelta(seconds=self.embargo_seconds),
            mode=self.mode,
        )


class Candidate(_Config):
    name: Name
    wrapper: Literal["baseline", "ridge"]
    features: Annotated[list[Name], Field(min_length=1)]
    missing: Literal["drop", "train_median"]
    ridge_penalty: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def validate_wrapper(self) -> Self:
        if len(set(self.features)) != len(self.features):
            raise ValueError("candidate features must be unique")
        if self.wrapper == "baseline" and (
            len(self.features) != 1 or self.ridge_penalty is not None
        ):
            raise ValueError("baseline requires one feature and no ridge_penalty")
        if self.wrapper == "ridge" and self.ridge_penalty is None:
            raise ValueError("ridge requires a positive ridge_penalty")
        return self


class Metrics(_Config):
    periods_per_year: Annotated[int, Field(ge=1, le=10_000_000)] = 252
    cost_bps: Annotated[float, Field(ge=0, le=10_000)] = 0
    volatility_lookback: Annotated[int, Field(ge=2, le=10_000)] = 5
    low_quantile: Annotated[float, Field(gt=0, lt=1)] = 1 / 3
    high_quantile: Annotated[float, Field(gt=0, lt=1)] = 2 / 3


class Selection(_Config):
    rule: Literal["validation_sharpe"] = "validation_sharpe"
    bootstrap_samples: Annotated[int, Field(ge=99, le=100_000)] = 1999
    block_length: Annotated[int, Field(ge=2, le=10_000)] = 5
    min_holdout_periods: Annotated[int, Field(ge=8)] = 40
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    seed: Seed


class RunConfig(_Config):
    schema_version: Annotated[int, Field(ge=1, le=1)]
    inputs: Inputs
    windows: Windows
    candidates: Annotated[list[Candidate], Field(min_length=1)]
    metrics: Metrics
    selection: Selection
    output_directory: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        columns = set(self.inputs.observation_columns)
        if len(columns) != len(self.inputs.observation_columns) or columns & RESERVED_COLUMNS:
            raise ValueError("observation_columns must be distinct and omit reserved columns")
        if self.inputs.price_column not in columns:
            raise ValueError("price_column must be an observation column")
        if len({candidate.name for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate names must be unique")
        for candidate in self.candidates:
            unknown = set(candidate.features) - columns - set(DERIVED_FEATURES)
            if unknown:
                raise ValueError(
                    f"candidate {candidate.name} has unknown features {sorted(unknown)}"
                )
        if self.metrics.low_quantile >= self.metrics.high_quantile:
            raise ValueError("low_quantile must be below high_quantile")
        self.windows.splitter()
        return self


class _UniqueKeysLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise ValueError("YAML mappings must have unique keys")
        return mapping


def load_config(path: Path) -> RunConfig:
    with path.open(encoding="utf-8") as stream:
        return RunConfig.model_validate(yaml.load(stream, Loader=_UniqueKeysLoader))
