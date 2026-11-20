# QuantCheck v1.0 release candidate

`1.0.0rc1` adds timestamp-level factor research through `pit-alpha run` and
the [Python API](PIT_ALPHA.md#python-api-and-evidence). It ships with a
seeded synthetic Parquet example, deterministic JSON and Markdown reports,
and a holdout Sharpe check. The supporting day-level audit remains available
through `quantcheck`. The immutable v0.1 release and v0.2 research evidence
keep their original versions and results.

The candidate supports Python 3.12. Its reference qualification uses Linux
with frozen uv dependencies, a locked Hatchling build, one OpenBLAS thread,
and two Polars threads. The fixture uses 24 symbols, 241 decision instants,
5,784 rows per input table, five trials, five OOS folds, and 150 OOS periods.
The timed workflow includes reading both Parquet inputs, research execution,
and writing both report formats.

The candidate also updates the locked Certifi trust bundle and the pinned
Python 3.12.16 Bookworm base to pick up security fixes detected on the
preceding revision.

## Candidate evidence

- [CI runs](https://github.com/rishixgamer/QuantCheck/actions/workflows/ci.yml)
  provide the qualification result and the `quantcheck-pit-alpha-candidate`
  artifact for each revision. The artifact contains the wheel and sdist,
  `SHA256SUMS`, the timed canary suite and benchmark records, and the two
  synthetic reports plus the installed CLI's day-level audit report.
- [Security runs](https://github.com/rishixgamer/QuantCheck/actions/workflows/security.yml)
  provide the matching dependency, secret, and container checks. Compare the
  run's checked-out revision with `candidate_revision` in the benchmark record.
- Acceptance tickets [#4](https://github.com/rishixgamer/QuantCheck/issues/4),
  [#5](https://github.com/rishixgamer/QuantCheck/issues/5),
  [#6](https://github.com/rishixgamer/QuantCheck/issues/6),
  [#7](https://github.com/rishixgamer/QuantCheck/issues/7),
  [#8](https://github.com/rishixgamer/QuantCheck/issues/8),
  [#9](https://github.com/rishixgamer/QuantCheck/issues/9), and
  [#10](https://github.com/rishixgamer/QuantCheck/issues/10) cover the code,
  scientific contracts, reports, and candidate qualification.

## Reproduce qualification

Run from the candidate checkout on Linux with uv and Python 3.12:

```bash
uv sync --frozen --all-groups
uv run python -m scripts.qualify_pit_alpha test --output dist/pit-alpha-tests.json
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build --offline --no-build-isolation
uv run python scripts/check_distribution_contents.py dist

candidate_dir="$(mktemp -d)"
uv venv --python 3.12 "$candidate_dir/venv"
uv export --locked --no-dev --no-emit-project --format requirements-txt > "$candidate_dir/requirements.txt"
uv pip sync --require-hashes --python "$candidate_dir/venv/bin/python" "$candidate_dir/requirements.txt"
uv pip install --python "$candidate_dir/venv/bin/python" --no-deps --offline dist/quantcheck-*.whl
uv run python -m scripts.qualify_pit_alpha package --python "$candidate_dir/venv/bin/python" --output dist
(cd dist && sha256sum *.whl *.tar.gz *.json *.md > SHA256SUMS && sha256sum --check SHA256SUMS)
```

The benchmark record captures the revision, execution environment, fixture
dimensions, trial and fold counts, elapsed time, peak RSS, and report schema.
The checks require the focused tests to finish in under 30 seconds and
the full reference run to finish in under 60 seconds at under 2 GB peak RSS.
The installed-candidate smoke check compares both reports byte for byte across
two runs, runs the existing audit CLI, and imports the three public factor APIs.
