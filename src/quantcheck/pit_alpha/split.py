"""Walk-forward windows with observable, non-overlapping fit and selection labels."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from quantcheck.pit_alpha.alignment import (
    DECISION_KEY,
    normalize_numbers,
    normalize_panel,
    require_unique,
)

LABEL_TIMESTAMPS = ("label_end_ts", "label_available_ts")


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    def contains(self) -> pl.Expr:
        return pl.col("decision_ts").is_between(self.start, self.end, closed="left")


@dataclass(frozen=True)
class Eligibility:
    window_rows: int
    embargoed: int
    purged: int
    unavailable: int
    missing_target: int
    eligible: int


@dataclass(frozen=True)
class Fold:
    number: int
    train_window: Window
    validation_window: Window
    oos_window: Window
    train: pl.DataFrame
    validation: pl.DataFrame
    oos: pl.DataFrame
    train_eligibility: Eligibility
    validation_eligibility: Eligibility


def _eligible(
    rows: pl.DataFrame, cutoff: datetime, embargo: timedelta
) -> tuple[pl.DataFrame, Eligibility]:
    outside_embargo = rows.filter(pl.col("decision_ts") < cutoff - embargo)
    # Null label timestamps are classified as unavailable, independently of purge.
    before_protected = outside_embargo.filter((pl.col("label_end_ts") < cutoff).fill_null(True))
    available = before_protected.filter(
        pl.col("label_end_ts").is_not_null() & (pl.col("label_available_ts") <= cutoff)
    )
    eligible = available.filter(pl.col("target_return").is_not_null())
    return eligible, Eligibility(
        window_rows=rows.height,
        embargoed=rows.height - outside_embargo.height,
        purged=outside_embargo.height - before_protected.height,
        unavailable=before_protected.height - available.height,
        missing_target=available.height - eligible.height,
        eligible=eligible.height,
    )


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Split by UTC durations with rolling or expanding training and disjoint OOS windows."""

    train: timedelta
    validation: timedelta
    oos: timedelta
    step: timedelta
    embargo: timedelta = timedelta(0)
    mode: Literal["rolling", "expanding"] = "rolling"

    def __post_init__(self) -> None:
        if min(self.train, self.validation, self.oos, self.step) <= timedelta(0):
            raise ValueError("train, validation, oos and step durations must be positive")
        if self.step < self.oos:
            raise ValueError("step must be at least the OOS duration to avoid duplicate OOS keys")
        if not timedelta(0) <= self.embargo < min(self.train, self.validation):
            raise ValueError("embargo must be nonnegative and shorter than train and validation")
        if self.mode not in {"rolling", "expanding"}:
            raise ValueError("window mode must be rolling or expanding")

    def split(self, labels: pl.DataFrame) -> Iterator[Fold]:
        """Yield complete usable folds, or raise when a planned fold has an empty role."""
        panel = normalize_numbers(
            normalize_panel(
                labels, timestamps=("decision_ts", *LABEL_TIMESTAMPS), nullable=LABEL_TIMESTAMPS
            ),
            ("target_return",),
        ).sort("decision_ts", "symbol")
        require_unique(panel, DECISION_KEY, "label panel")
        if panel.select((pl.col("target_return") < -1).any()).item():
            raise ValueError("simple asset returns cannot be below -1")
        if not panel.height:
            raise ValueError("label panel is empty")
        if panel.select(
            (
                (pl.col("label_end_ts") <= pl.col("decision_ts"))
                | (pl.col("label_available_ts") < pl.col("label_end_ts"))
                | (pl.col("label_end_ts").is_null() & pl.col("label_available_ts").is_not_null())
            ).any()
        ).item():
            raise ValueError("labels require decision_ts < label_end_ts <= label_available_ts")
        anchor: datetime = panel.get_column("decision_ts")[0]
        last: datetime = panel.get_column("decision_ts")[-1]
        number = 1
        validation_start = anchor + self.train
        while validation_start + self.validation + self.oos <= last:
            train_start = anchor if self.mode == "expanding" else validation_start - self.train
            oos_start = validation_start + self.validation
            train_window = Window(train_start, validation_start)
            validation_window = Window(validation_start, oos_start)
            oos_window = Window(oos_start, oos_start + self.oos)
            train, train_counts = _eligible(
                panel.filter(train_window.contains()), validation_start, self.embargo
            )
            validation, validation_counts = _eligible(
                panel.filter(validation_window.contains()), oos_start, self.embargo
            )
            oos = panel.filter(oos_window.contains())
            if min(train.height, validation.height, oos.height) == 0:
                raise ValueError(
                    f"fold {number} has an empty usable partition "
                    f"(train={train.height}, validation={validation.height}, oos={oos.height})"
                )
            yield Fold(
                number,
                train_window,
                validation_window,
                oos_window,
                train,
                validation,
                oos,
                train_counts,
                validation_counts,
            )
            number += 1
            validation_start += self.step
        if number == 1:
            raise ValueError("label panel does not span a complete walk-forward fold")
