# pit-alpha

**Test your signal with the data you actually had.** A point-in-time factor research engine with purged walk-forward validation, fold-local fitting, and an auditable post-selection Sharpe check.

[Try the example](#quickstart) · [Data contract, CLI, and Python API](docs/PIT_ALPHA.md) · [v1.0 release notes](docs/RELEASE_NOTES_1.0.md) · [Project site](https://rishixgamer.github.io/pit-alpha/)

![pit-alpha. Diagram of staggered training, validation, and out-of-sample windows with embargo gaps.](docs/assets/pit-alpha-cover.svg)

## Quickstart

Python 3.12 · uv · Polars/Parquet input.

```sh
git clone https://github.com/rishixgamer/pit-alpha.git
cd pit-alpha
uv sync --frozen --all-groups
uv run pit-alpha example --directory ./example-run
uv run pit-alpha run --config ./example-run/example.yaml
```

The seeded example writes Markdown and JSON reports to `example-run/reports` with fold-by-fold metrics, regime breakdowns, and selection evidence.

## What you can do

- **Align.** Join Parquet observations by symbol and timestamp. Only revisions available at the decision time can enter features.
- **Validate.** Purge overlapping labels, enforce an embargo, and fit preprocessing and models inside each fold. A future-price canary exercises the full path.
- **Evaluate.** Inspect IC, Rank IC, ICIR, Sharpe, drawdown, turnover, and volatility regimes. Review the candidate search and adjusted holdout Sharpe check.

## Reproducible evidence

CI installs the wheel and source distribution in separate Python 3.12 environments. Each runs the same seeded example with 24 synthetic symbols, 5,784 rows per input table, five trials, and five out-of-sample folds. The reports must match byte for byte. See the [qualification commands](docs/RELEASE_NOTES_1.0.md#reproduce-qualification) and [CI artifacts](https://github.com/rishixgamer/pit-alpha/actions/workflows/ci.yml).

## QuantCheck research archive

`pit-alpha` grew out of QuantCheck. The day-level data audit remains available as `quantcheck`. The distribution is still `quantcheck` and exposes `asof_align`, `PurgedWalkForwardSplit`, and `evaluate_signals` under `quantcheck.pit_alpha`. See the [original audit quickstart, frozen benchmark, and evidence](docs/QUANTCHECK.md).

---

[GitHub Releases](https://github.com/rishixgamer/pit-alpha/releases) · [MIT License](LICENSE) · [Issues](https://github.com/rishixgamer/pit-alpha/issues)

If pit-alpha helps your research, [star the repository](https://github.com/rishixgamer/pit-alpha).
