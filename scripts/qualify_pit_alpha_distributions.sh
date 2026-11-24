#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-dist}"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
export SOURCE_DATE_EPOCH
uv build --offline --no-build-isolation --out-dir "$output_dir"
uv run --locked python scripts/check_distribution_contents.py "$output_dir"

uv export --locked --no-dev --no-emit-project --format requirements-txt \
  --quiet --output-file "$scratch_dir/runtime-requirements.txt"
uv export --locked --only-group dev --format requirements-txt \
  --quiet --output-file "$scratch_dir/build-constraints.txt"

qualify() {
  local kind="$1" distribution="$2" python="$scratch_dir/$1/bin/python"
  uv venv --python 3.12 "$scratch_dir/$kind"
  uv pip sync --require-hashes --python "$python" "$scratch_dir/runtime-requirements.txt"
  uv pip install --no-deps --build-constraints "$scratch_dir/build-constraints.txt" \
    --python "$python" "$distribution"
  uv run --locked python -m scripts.qualify_pit_alpha package \
    --python "$python" --output "$scratch_dir/$kind-evidence"
}
qualify wheel "$output_dir"/quantcheck-*.whl
qualify sdist "$output_dir"/quantcheck-*.tar.gz

for report in pit-alpha-report.json pit-alpha-report.md day-level-audit-report.json; do
  cmp "$scratch_dir/wheel-evidence/$report" "$scratch_dir/sdist-evidence/$report"
done
cp "$scratch_dir/wheel-evidence/"* "$output_dir/"
cp "$scratch_dir/sdist-evidence/pit-alpha-benchmark.json" "$output_dir/pit-alpha-sdist-benchmark.json"
(
  cd "$output_dir"
  sha256sum ./*.whl ./*.tar.gz ./*.json ./*.md > SHA256SUMS
  sha256sum --check SHA256SUMS
)
