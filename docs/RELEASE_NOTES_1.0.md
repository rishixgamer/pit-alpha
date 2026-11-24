# pit-alpha v1.0.0

`1.0.0` adds timestamp-level factor research through `pit-alpha run` and
the [Python API](PIT_ALPHA.md#python-api-and-evidence). It ships with a
seeded synthetic Parquet example, deterministic JSON and Markdown reports,
and a holdout Sharpe check. The `quantcheck` distribution provides both
`pit-alpha` and the existing `quantcheck` day-level audit CLI. The example
measures the workflow on synthetic data.

The supported runtime is Python 3.12. Reference qualification uses Linux
with uv 0.12.2, frozen dependencies, a locked Hatchling build, one OpenBLAS thread,
and two Polars threads. The fixture uses 24 symbols, 241 decision instants,
5,784 rows per input table, five trials, five OOS folds, and 150 OOS periods.
The timed workflow includes reading both Parquet inputs, research execution,
and writing both report formats.

## Install

Download `quantcheck-1.0.0-py3-none-any.whl` and `SHA256SUMS` from the
[`v1.0.0` GitHub release](https://github.com/rishixgamer/pit-alpha/releases).
Run these commands from the download directory with a Python 3.12 virtual
environment active.

```bash
sha256sum --ignore-missing --check SHA256SUMS && \
  python -m pip install ./quantcheck-1.0.0-py3-none-any.whl
pit-alpha example --directory ./example-run
pit-alpha run --config ./example-run/example.yaml
```

## Release evidence

- The [CI workflow](https://github.com/rishixgamer/pit-alpha/actions/workflows/ci.yml)
  produces `quantcheck-pit-alpha-candidate` for each successful run. It contains
  the wheel and sdist, `SHA256SUMS`, a timed unit and canary result, separate
  wheel and sdist benchmark records, both synthetic reports, and a day-level
  audit report.
- Match `candidate_revision` in the benchmark records to the source commit in
  the [Security workflow](https://github.com/rishixgamer/pit-alpha/actions/workflows/security.yml)
  for dependency, secret, and container verification.
- Acceptance tickets [#4](https://github.com/rishixgamer/pit-alpha/issues/4),
  [#5](https://github.com/rishixgamer/pit-alpha/issues/5),
  [#6](https://github.com/rishixgamer/pit-alpha/issues/6),
  [#7](https://github.com/rishixgamer/pit-alpha/issues/7),
  [#8](https://github.com/rishixgamer/pit-alpha/issues/8),
  [#9](https://github.com/rishixgamer/pit-alpha/issues/9), and
  [#10](https://github.com/rishixgamer/pit-alpha/issues/10) cover the code,
  scientific contracts, reports, and candidate qualification.

## Reproduce qualification

Run from the release checkout on Linux with uv 0.12.2 and Python 3.12. Use an
empty `dist/` directory.

```bash
uv sync --frozen --all-groups
uv lock --check
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src tests
uv run --locked python -m scripts.qualify_pit_alpha test --output dist/pit-alpha-tests.json
uv run --locked pytest
bash scripts/qualify_pit_alpha_distributions.sh dist
```

The benchmark record captures the revision, execution environment, fixture
dimensions, trial and fold counts, elapsed time, peak RSS, and report schema.
The checks require the focused tests to finish in under 30 seconds and each
full reference run to finish in under 60 seconds at under 2 GB peak RSS. Each
clean install reproduces both factor reports, runs the existing audit CLI,
and imports all three public factor APIs. Both installs must produce identical
reports.

## Prepare the GitHub release

After merging, verify that the CI and Security push runs both passed for the
same `main` commit. Create `v1.0.0` at that exact revision. Include the CI run,
Security run, and qualification artifact links in the release description.
The [publication workflow](https://github.com/rishixgamer/pit-alpha/actions/workflows/publish-self-hosted.yml)
uses the matching CI evidence, rebuilds the wheel and sdist, and checks them
byte for byte before uploading them with the benchmark records, reports,
checksums, and SBOMs.

Rerun the original tag-triggered workflow after a transient failure if there are
no release attachments. If the tagged workflow needs a code change, recover the
files from the original qualified build and verify the existing attestations. If
the release already has attachments, verify their checksums and recover only the
missing files.
