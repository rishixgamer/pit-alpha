from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quantcheck.pit_alpha import asof_align
from quantcheck.pit_alpha.alignment import UTC_TIMESTAMP

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def instant(microseconds: int) -> datetime:
    return ORIGIN + timedelta(microseconds=microseconds)


def observations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["X", "Y", "X", "X", "X", "X", "X"],
            "event_ts": [instant(i) for i in (3, 1, 2, 2, 1, 3, 4)],
            "available_ts": [
                instant(8),
                instant(1),
                instant(5),
                instant(2),
                instant(5),
                instant(5),
                None,
            ],
            "signal": [31.0, 100.0, 21.0, 20.0, 11.0, 30.0, 999.0],
        },
        schema_overrides={"event_ts": UTC_TIMESTAMP, "available_ts": UTC_TIMESTAMP},
    )


def decisions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["X", "Z", "Y", "X", "X", "X", "X"],
            "decision_ts": [instant(i) for i in (8, 5, 5, 7, 1, 2, 5)],
        },
        schema_overrides={"decision_ts": UTC_TIMESTAMP},
    )


def test_latest_event_then_latest_revision_at_inclusive_observation_time(tmp_path: Path) -> None:
    source, schedule = observations(), decisions()
    original_source, original_schedule = source.clone(), schedule.clone()
    result = asof_align(source, schedule, value_columns=["signal"])
    assert result["symbol"].to_list() == ["X", "X", "X", "Y", "Z", "X", "X"]
    assert result["signal"].to_list() == [None, 20, 30, 100, None, 30, 31]
    assert result["available_ts"].to_list() == [
        None,
        instant(2),
        instant(5),
        instant(1),
        None,
        instant(5),
        instant(8),
    ]
    source.write_parquet(tmp_path / "observations.parquet")
    schedule.write_parquet(tmp_path / "decisions.parquet")
    from_paths = asof_align(
        tmp_path / "observations.parquet", tmp_path / "decisions.parquet", value_columns=["signal"]
    )
    assert_frame_equal(from_paths, result, check_exact=True)
    assert_frame_equal(source, original_source, check_exact=True)
    assert_frame_equal(schedule, original_schedule, check_exact=True)
    assert_frame_equal(
        asof_align(source.reverse(), schedule.reverse(), value_columns=["signal"]),
        result,
        check_exact=True,
    )


def test_revision_updates_until_a_newer_event_is_observable() -> None:
    source = observations().filter(pl.col("event_ts") < instant(3))
    result = asof_align(source, decisions(), value_columns=["signal"])
    assert result.filter(pl.col("symbol") == "X")["signal"].to_list() == [None, 20, 21, 21, 21]


def test_timestamp_normalization_and_null_source_values() -> None:
    source = observations().with_columns(
        pl.col("available_ts").dt.convert_time_zone("America/New_York"),
        pl.lit(float("nan")).alias("signal"),
    )
    result = asof_align(source, decisions(), value_columns=["signal"])
    assert result["available_ts"].dtype == UTC_TIMESTAMP
    assert result["signal"].null_count() == result.height
    assert result["event_ts"].null_count() == 2
    empty = source.with_columns(pl.lit(None, dtype=UTC_TIMESTAMP).alias("available_ts"))
    unmatched = asof_align(empty, decisions(), value_columns=["signal"])
    assert unmatched["event_ts"].null_count() == unmatched.height


@pytest.mark.parametrize(
    ("make_source", "match"),
    [
        (lambda source: pl.concat([source, source.head(1)]), "observations has duplicate"),
        (
            lambda source: source.with_columns(pl.lit(instant(0)).alias("available_ts")),
            "cannot precede event_ts",
        ),
        (
            lambda source: source.with_columns(pl.col("event_ts").dt.replace_time_zone(None)),
            "timezone-aware",
        ),
        (
            lambda source: source.with_columns(pl.col("event_ts").dt.cast_time_unit("ns")),
            "microsecond",
        ),
    ],
)
def test_rejects_ambiguous_or_invalid_observations(
    make_source: Callable[[pl.DataFrame], pl.DataFrame], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        asof_align(make_source(observations()), decisions(), value_columns=["signal"])


def test_rejects_duplicate_decisions() -> None:
    schedule = decisions()
    with pytest.raises(ValueError, match="decisions has duplicate"):
        asof_align(observations(), pl.concat([schedule, schedule]), value_columns=["signal"])
