"""Point-in-time selection from a timestamped, symbol-keyed observation panel."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

Panel = pl.DataFrame | str | Path

DECISION_KEY = ("symbol", "decision_ts")
OBSERVATION_KEY = ("symbol", "event_ts", "available_ts")
UTC_TIMESTAMP = pl.Datetime("us", "UTC")


def read_panel(panel: Panel) -> pl.DataFrame:
    return panel if isinstance(panel, pl.DataFrame) else pl.read_parquet(panel)


def normalize_panel(
    frame: pl.DataFrame,
    *,
    timestamps: Sequence[str],
    nullable: Sequence[str] = (),
) -> pl.DataFrame:
    """Validate common keys and normalize timezone-aware millisecond/microsecond instants."""
    if frame.schema.get("symbol") != pl.String:
        raise ValueError("symbol must be a string column")
    if (
        frame.select(pl.col("symbol").is_null() | (pl.col("symbol").str.strip_chars() == ""))
        .to_series()
        .any()
    ):
        raise ValueError("symbol cannot be null or blank")
    for name in timestamps:
        dtype = frame.schema.get(name)
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None or dtype.time_unit == "ns":
            raise ValueError(
                f"{name} must be a timezone-aware millisecond or microsecond timestamp"
            )
        if name not in nullable and frame.get_column(name).null_count():
            raise ValueError(f"{name} cannot be null")
    return frame.with_columns(pl.col(name).cast(UTC_TIMESTAMP) for name in timestamps)


def require_unique(frame: pl.DataFrame, key: Sequence[str], description: str) -> None:
    if frame.select(pl.struct(*key).is_duplicated().any()).item():
        raise ValueError(f"{description} has duplicate keys ({', '.join(key)})")


def normalize_numbers(frame: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    for name in names:
        dtype = frame.schema.get(name)
        if dtype is None or not (dtype.is_numeric() or dtype == pl.Null):
            raise ValueError(f"{name} must be a numeric column")
    result = frame.with_columns(pl.col(name).cast(pl.Float64).fill_nan(None) for name in names)
    if (
        names
        and result.select(
            pl.any_horizontal(pl.col(name).is_infinite() for name in names).any()
        ).item()
    ):
        raise ValueError(f"infinite values in numeric columns ({', '.join(names)})")
    return result


def asof_align(
    observations: Panel,
    decisions: Panel,
    *,
    value_columns: Sequence[str],
) -> pl.DataFrame:
    """Select the latest known event, then its latest known revision, for each decision.

    Return decisions ordered by UTC instant and symbol, with null observation
    fields for unmatched keys. Each row's event and availability identify the
    actual source revision selected.
    """
    columns = tuple(value_columns)
    if (
        not columns
        or len(set(columns)) != len(columns)
        or set(columns)
        & {
            *OBSERVATION_KEY,
            *DECISION_KEY,
        }
    ):
        raise ValueError("value_columns must be distinct numeric observation fields")
    queries = normalize_panel(read_panel(decisions), timestamps=("decision_ts",)).select(
        *DECISION_KEY
    )
    require_unique(queries, DECISION_KEY, "decisions")
    sources = normalize_panel(
        read_panel(observations),
        timestamps=("event_ts", "available_ts"),
        nullable=("available_ts",),
    ).filter(pl.col("available_ts").is_not_null())
    if sources.select((pl.col("available_ts") < pl.col("event_ts")).any()).item():
        raise ValueError("observation available_ts cannot precede event_ts")
    sources = normalize_numbers(sources.select(*OBSERVATION_KEY, pl.col(columns)), columns)
    require_unique(sources, OBSERVATION_KEY, "observations")

    # An arrival can update the snapshot only if it reaches or revises the event
    # frontier. Orders within an arrival time are explicit, even with mixed symbols.
    frontier = (
        sources.sort("symbol", "available_ts", "event_ts")
        .filter(pl.col("event_ts") == pl.col("event_ts").cum_max().over("symbol"))
        .unique(subset=["symbol", "available_ts"], keep="last")
        .sort("available_ts")
    )
    return queries.sort("decision_ts", "symbol").join_asof(
        frontier,
        left_on="decision_ts",
        right_on="available_ts",
        by="symbol",
        check_sortedness=False,
    )
