from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quantcheck.pit_alpha import PurgedWalkForwardSplit
from quantcheck.pit_alpha.alignment import UTC_TIMESTAMP
from quantcheck.pit_alpha.split import Eligibility

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def instant(seconds: int) -> datetime:
    return ORIGIN + timedelta(seconds=seconds)


def panel() -> pl.DataFrame:
    rows = []
    for time in (0, 2, 4, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 25, 26, 30, 34):
        for symbol in ("X", "Y"):
            end = {2: 10, 4: 11, 12: 18}.get(time, time + 1)
            available = {5: 10, 7: 11, 15: 18}.get(time, end)
            target: float | None = 0.01
            if time == 7 and symbol == "Y":
                available = 9
                target = None
            rows.append((symbol, instant(time), instant(end), instant(available), target))
    return pl.DataFrame(
        list(reversed(rows)),
        schema={
            "symbol": pl.String,
            "decision_ts": UTC_TIMESTAMP,
            "label_end_ts": UTC_TIMESTAMP,
            "label_available_ts": UTC_TIMESTAMP,
            "target_return": pl.Float64,
        },
        orient="row",
    )


def test_timestamp_windows_purge_embargo_and_label_availability() -> None:
    split = PurgedWalkForwardSplit(
        train=timedelta(seconds=10),
        validation=timedelta(seconds=8),
        oos=timedelta(seconds=8),
        step=timedelta(seconds=8),
        embargo=timedelta(seconds=2),
    )
    first, second = split.split(panel())
    assert (first.train_window.start, first.train_window.end) == (instant(0), instant(10))
    assert (first.oos_window.start, first.oos_window.end) == (instant(18), instant(26))
    assert first.train_eligibility == Eligibility(14, 4, 4, 1, 1, 4)
    assert first.validation_eligibility == Eligibility(10, 4, 2, 0, 0, 4)
    assert first.train["decision_ts"].to_list() == [instant(0)] * 2 + [instant(5)] * 2
    assert first.validation["decision_ts"].to_list() == [instant(10)] * 2 + [instant(15)] * 2
    assert first.oos["decision_ts"].to_list() == [
        instant(time) for time in (18, 22, 25) for _ in range(2)
    ]
    assert second.oos["decision_ts"].to_list() == [
        instant(time) for time in (26, 30) for _ in range(2)
    ]
    assert second.train_window.start == instant(8)
    expanding = PurgedWalkForwardSplit(
        split.train, split.validation, split.oos, split.step, split.embargo, "expanding"
    )
    assert tuple(expanding.split(panel()))[1].train_window.start == instant(0)


def test_rejects_overlapping_invalid_and_empty_windows() -> None:
    eight, ten = timedelta(seconds=8), timedelta(seconds=10)
    with pytest.raises(ValueError, match="duplicate OOS keys"):
        PurgedWalkForwardSplit(ten, eight, eight, timedelta(seconds=7))
    with pytest.raises(ValueError, match="embargo"):
        PurgedWalkForwardSplit(ten, eight, eight, eight, eight)
    with pytest.raises(ValueError, match="positive"):
        PurgedWalkForwardSplit(timedelta(0), eight, eight, eight)
    splitter = PurgedWalkForwardSplit(ten, eight, eight, eight)
    with pytest.raises(ValueError, match="complete walk-forward fold"):
        list(splitter.split(panel().filter(pl.col("decision_ts") < instant(20))))
    with pytest.raises(ValueError, match="empty usable partition"):
        list(
            splitter.split(
                panel().filter(
                    ~pl.col("decision_ts").is_between(instant(10), instant(17), closed="both")
                )
            )
        )
    with pytest.raises(ValueError, match="labels require"):
        list(splitter.split(panel().with_columns(pl.col("decision_ts").alias("label_end_ts"))))
    with pytest.raises(ValueError, match="duplicate keys"):
        list(splitter.split(pl.concat([panel(), panel().head(1)])))
