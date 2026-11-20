"""Render both report formats from one result and publish them together."""

import json
import os
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

from quantcheck import __version__
from quantcheck.pit_alpha.metrics import Statistic, Summary
from quantcheck.pit_alpha.workflow import REPORT_SCHEMA_VERSION, SELECTION_RULE, Run


def _table(head: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
        *(
            "| " + " | ".join(str(value).replace("|", r"\|") for value in row) + " |"
            for row in rows
        ),
        "",
    ]


def _statistic(statistic: Statistic) -> str:
    if statistic.value is None:
        return f"Undefined ({statistic.count}, {statistic.reason})"
    return f"{statistic.value} ({statistic.count})"


def _summary_table(run: Run) -> list[str]:
    groups = [("Combined", run.metrics.combined)]
    groups.extend((f"Fold {fold}", summary) for fold, summary in run.metrics.by_fold.items())
    groups.extend(run.metrics.by_regime.items())
    for fold, summaries in run.metrics.by_fold_regime.items():
        groups.extend((f"Fold {fold} / {regime}", summary) for regime, summary in summaries.items())
    names = tuple(field.name for field in fields(Summary) if field.type is Statistic)
    return _table(
        ("Group", "Decisions", "Scored", "Paired", *names),
        [
            (
                name,
                summary.decisions,
                summary.predictions,
                summary.paired_observations,
                *(_statistic(getattr(summary, field)) for field in names),
            )
            for name, summary in groups
        ],
    )


def markdown(run: Run, input_hashes: dict[str, str]) -> str:
    lines = [
        "# QuantCheck pit-alpha",
        "",
        f"Report schema {REPORT_SCHEMA_VERSION} · QuantCheck {__version__}",
        "",
        "## Availability and folds",
        "",
        "All timestamps are UTC microseconds. Observations require "
        "`event_ts <= available_ts <= decision_ts`. The latest observable event wins, "
        "followed by the latest observable revision. Null and NaN are treated as missing.",
        "",
        f"{run.observations} source observations and {run.decisions} decisions produced "
        f"{run.aligned_rows} aligned and {run.unmatched_rows} unmatched rows.",
        "",
        "Windows are half-open and OOS keys are disjoint. Fit and selection require "
        "`decision_ts < cutoff - embargo`, `label_end_ts < cutoff` and "
        "`label_available_ts <= cutoff`. Cutoffs are the validation start for fitting "
        "and the first OOS start for selection.",
        "",
        *[f"- {name} SHA-256 `{digest}`" for name, digest in input_hashes.items()],
        "",
    ]
    lines += _table(
        (
            "Fold",
            "Train",
            "Validation",
            "OOS",
            "Train eligible",
            "Validation eligible",
            "OOS rows",
            "OOS unmatched",
            "OOS missing predictions",
        ),
        [
            (
                fold.number,
                *(
                    f"[{window['start']}, {window['end_exclusive']})"
                    for window in fold.windows.values()
                ),
                fold.train.eligible,
                fold.validation.eligible,
                fold.oos_rows,
                fold.oos_unmatched,
                fold.oos_missing_predictions,
            )
            for fold in run.folds
        ],
    )
    lines += _table(
        ("Fold", "Role", "Window rows", "Embargoed", "Purged", "Unavailable", "Missing target"),
        [
            (
                fold.number,
                role,
                counts.window_rows,
                counts.embargoed,
                counts.purged,
                counts.unavailable,
                counts.missing_target,
            )
            for fold in run.folds
            for role, counts in (("Train", fold.train), ("Validation", fold.validation))
        ],
    )
    lines += [
        "## Candidate selection and post-selection Sharpe",
        "",
        f"Selection used {SELECTION_RULE}. Candidate `{run.selected.name}` "
        f"(`{run.selected.candidate_id}`) was frozen before the first OOS window.",
        "",
    ]
    lines += _table(
        (
            "Candidate",
            "Configuration SHA-256",
            "Status",
            "Validation Sharpe",
            "Observed periods",
            "Reason or duplicate",
        ),
        [
            (
                trial.name,
                trial.candidate_id,
                trial.status,
                trial.validation_sharpe,
                trial.validation_periods,
                trial.reason or trial.duplicate_of or "",
            )
            for trial in run.trials
        ],
    )
    check = run.sharpe_check
    lines += _table(
        (
            "Method",
            "Status",
            "Reason",
            "Trials",
            "Holdout periods",
            "Observed periods",
            "Omitted decisions",
            "Raw Sharpe",
            "Raw p",
            "Adjusted p",
            "Minimum adjusted p",
            "Passes",
        ),
        [
            (
                check.method,
                check.status,
                check.reason,
                check.attempted_trials,
                check.holdout_periods,
                check.observed_periods,
                check.omitted_decisions,
                check.raw_sharpe,
                check.raw_p,
                check.adjusted_p,
                check.minimum_adjusted_p,
                check.passes,
            )
        ],
    )
    selection = run.configuration["selection"]
    lines += [
        f"Bootstrap seed {selection['seed']}, {selection['bootstrap_samples']} replicates, "
        f"block length {selection['block_length']}, minimum {selection['min_holdout_periods']} "
        f"holdout periods, alpha {selection['alpha']}.",
        "",
        "## Out-of-sample metrics",
        "",
        "Each metric includes its sample count in parentheses. See the JSON report for "
        "individual period returns, fitted model parameters, missingness and regime thresholds.",
        "",
    ]
    lines += _summary_table(run)
    return "\n".join(lines)


def write_reports(directory: Path, run: Run, input_hashes: dict[str, str]) -> None:
    contents = {
        "report.json": json.dumps(
            run.report(input_hashes), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        "report.md": markdown(run, input_hashes),
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pit-alpha-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        for name, content in contents.items():
            (staging / name).write_bytes(content.encode("utf-8"))
        try:
            os.replace(staging, directory)
        except OSError:
            if directory.is_symlink() or not directory.is_dir():
                raise
            if {path.name for path in directory.iterdir()} != contents.keys() or any(
                not (directory / name).is_file()
                or (directory / name).read_bytes() != content.encode("utf-8")
                for name, content in contents.items()
            ):
                raise ValueError(
                    f"report directory already contains different results: {directory}"
                ) from None
