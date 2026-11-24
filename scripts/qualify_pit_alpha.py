"""Verify a locked build and record Linux release-candidate evidence."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from tests.cli_helpers import fault_case, write_case_config

from quantcheck.pit_alpha.example import EXAMPLE_DATES, EXAMPLE_SEED, EXAMPLE_SYMBOLS, write_example

TEST_LIMIT_SECONDS = 30
WALK_FORWARD_LIMIT_SECONDS = 60
WALK_FORWARD_LIMIT_BYTES = 2_000_000_000
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "2"}

# A dedicated process measures only the benchmark CLI child. On Linux ru_maxrss
# is in KiB. The example generation and package checks run outside this timer.
_MEASURER = """
import json
import resource
import subprocess
import sys
from time import perf_counter

start = perf_counter()
subprocess.run(sys.argv[1:], check=True, timeout=60, stdout=sys.stderr)
print(json.dumps({
    "elapsed_seconds": perf_counter() - start,
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
}))
"""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return platform.machine()


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu": _cpu(),
        "available_logical_cpus": len(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "threads": THREADS,
    }


def _tests(output: Path) -> None:
    paths = sorted(str(path) for path in Path("tests").glob("test_pit_alpha_*.py"))
    if not paths:
        raise ValueError("run qualification from the repository root")
    command = [sys.executable, "-m", "pytest", "-q", *paths]
    start = perf_counter()
    result = subprocess.run(
        command,
        timeout=TEST_LIMIT_SECONDS,
        capture_output=True,
        text=True,
        env=os.environ | THREADS,
    )
    elapsed = perf_counter() - start
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    if elapsed >= TEST_LIMIT_SECONDS:
        raise ValueError("pit-alpha tests must finish in under 30 seconds")
    _write_json(
        output,
        {
            "candidate_revision": _revision(),
            "command": ["python", *command[1:]],
            "elapsed_seconds": elapsed,
            "limit_seconds": TEST_LIMIT_SECONDS,
            "environment": _environment(),
            "pytest_stdout": result.stdout,
        },
    )


def _package(python: Path, output: Path) -> None:
    bin_dir = python.absolute().parent
    python = bin_dir / python.name
    pit_alpha = str(bin_dir / "pit-alpha")
    audit = str(bin_dir / "quantcheck")
    environment = os.environ | THREADS
    expected_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    with TemporaryDirectory(prefix="quantcheck-qualification-") as location:
        work = Path(location)
        write_example(work / "example")
        command = [pit_alpha, "run", "--config", "example/example.yaml"]
        measured = subprocess.run(
            [sys.executable, "-I", "-c", _MEASURER, *command],
            cwd=work,
            env=environment,
            check=True,
            timeout=65,
            capture_output=True,
            text=True,
        )
        timing = json.loads(measured.stdout)
        print(measured.stderr, end="")
        if timing["elapsed_seconds"] >= WALK_FORWARD_LIMIT_SECONDS:
            raise ValueError("walk-forward run must finish in under 60 seconds")
        if timing["peak_rss_bytes"] >= WALK_FORWARD_LIMIT_BYTES:
            raise ValueError("walk-forward process must use under 2 GB peak RSS")

        paths = [work / "example" / "reports" / name for name in ("report.json", "report.md")]
        first = [path.read_bytes() for path in paths]
        document = json.loads(first[0])
        combined = document["metrics"]["combined"]
        check = document["selection"]["check"]
        if (
            document["schema_version"] != 1
            or document["quantcheck_version"] != expected_version
            or len(document["folds"]) != 5
            or combined["decisions"] != 150
            or not check["passes"]
            or str(check["adjusted_p"]).encode() not in first[1]
            or str(combined["ic"]["value"]).encode() not in first[1]
        ):
            raise ValueError("installed example did not emit the expected matching reports")

        subprocess.run(
            [pit_alpha, "example", "--directory", "repeat"],
            cwd=work,
            env=environment,
            check=True,
            timeout=30,
        )
        subprocess.run(
            [pit_alpha, "run", "--config", "repeat/example.yaml"],
            cwd=work,
            env=environment | {"PYTHONHASHSEED": "813"},
            check=True,
            timeout=60,
        )
        if [work.joinpath("repeat", "reports", path.name).read_bytes() for path in paths] != first:
            raise ValueError("installed example did not reproduce both reports byte for byte")

        write_case_config(work / "case.json", fault_case("lookahead_timestamp"))
        subprocess.run(
            [audit, "inject", "--case", "case.json", "--output", "audit"],
            cwd=work,
            env=environment,
            check=True,
            timeout=30,
        )
        subprocess.run(
            [audit, "audit", "--dir", "audit"],
            cwd=work,
            env=environment,
            check=True,
            timeout=30,
        )
        (audit_report,) = work.glob("audit/public/cases/*/audit_report.json")
        installed = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import json, platform\n"
                "from importlib.metadata import version\n"
                "from quantcheck import __version__, pit_alpha\n"
                "for name in ('asof_align', 'PurgedWalkForwardSplit', 'evaluate_signals'):\n"
                "    assert callable(getattr(pit_alpha, name))\n"
                "assert __version__ == version('quantcheck') == "
                f"{expected_version!r}\n"
                "print(json.dumps({'python': platform.python_version()}))",
            ],
            cwd=work,
            env=environment,
            timeout=30,
            text=True,
        )

        output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copyfile(path, output / f"pit-alpha-{path.name}")
        shutil.copyfile(audit_report, output / "day-level-audit-report.json")
        _write_json(
            output / "pit-alpha-benchmark.json",
            {
                "candidate_revision": _revision(),
                "candidate_version": document["quantcheck_version"],
                "command": ["pit-alpha", *command[1:]],
                "environment": _environment() | json.loads(installed),
                "fixture": {
                    "seed": EXAMPLE_SEED,
                    "symbols": EXAMPLE_SYMBOLS,
                    "decision_instants": EXAMPLE_DATES,
                    "observations": document["availability"]["observations"],
                    "decisions": document["availability"]["decisions"],
                    "input_sha256": document["input_sha256"],
                },
                "candidate_trials": document["selection"]["attempted"],
                "folds": len(document["folds"]),
                "out_of_sample_periods": combined["decisions"],
                "report_schema_version": document["schema_version"],
                "reproduced_both_reports": True,
                "installed_audit_and_factor_apis": True,
                "limits": {
                    "elapsed_seconds": WALK_FORWARD_LIMIT_SECONDS,
                    "peak_rss_bytes": WALK_FORWARD_LIMIT_BYTES,
                },
                **timing,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tests = commands.add_parser("test")
    tests.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--python", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "test":
        _tests(args.output)
    else:
        _package(args.python, args.output)


if __name__ == "__main__":
    main()
