# pit-alpha

Pit-alpha runs timestamped factor research on explicitly available market data.
It uses Polars and Parquet, fold-local model fitting, and an auditable separation
between candidate selection and out-of-sample (OOS) evaluation.

## Quickstart

With Python 3.12 and uv, from a source checkout:

```bash
uv sync --frozen --all-groups
uv run pit-alpha example --directory /tmp/quantcheck-pit-alpha-example
uv run pit-alpha run --config /tmp/quantcheck-pit-alpha-example/example.yaml
```

The example command generates two seeded Parquet inputs and copies the
[bundled version 1 YAML config](../src/quantcheck/pit_alpha/example.yaml). The
run command writes `reports/report.json` and `reports/report.md` relative to
the config's directory. A wheel installation provides the same `pit-alpha`
commands without `uv run`. It also provides `quantcheck` for the day-level
audit workflow.

Relative input and output paths resolve from the YAML file. Use a fresh or empty
report directory. Repeating an identical run against the same directory succeeds
when both reports are byte-identical. A different result requires a different
output directory. An invalid run exits with a nonzero status and diagnostic
before it publishes either report.

[Weekly crypto momentum study](../examples/research/RESULTS.md) uses historical prices and reports the result after costs and multiple-trial adjustment.

## Input contracts

Both Parquet tables use a nonblank string `symbol`. Timestamp columns must be
timezone-aware and stored at millisecond or microsecond precision. The workflow
normalizes them to UTC microseconds; it rejects naive and nanosecond timestamps
so the conversion cannot silently round decision boundaries.

| Table | Columns | Meaning |
| --- | --- | --- |
| Observations | `symbol`, `event_ts`, `available_ts`, explicit numeric observation columns | Event time describes the observation. Availability identifies when that revision first became knowable. Null availability excludes the row. |
| Decisions and labels | `symbol`, `decision_ts`, `label_end_ts`, `label_available_ts`, numeric `target_return` | A unique decision key has a forward simple asset return, a label interval ending strictly after the decision, and availability at or after its end. Missing target or label timestamps make the outcome ineligible. |

A decision sees only same-symbol rows with `event_ts <= available_ts <= decision_ts`.
At the inclusive cutoff, choose the maximum visible event time, then the latest
visible revision of that event. Two visible rows with the same symbol, event
time, and availability are ambiguous and the input is rejected. A late revision
of an older event cannot displace a newer observable event. Output retains the
matched timestamps, and an unmatched decision retains its key with null source
columns. The aligned table projects the configured numeric observation columns
and preserves caller-owned inputs. Input order has no effect.

The YAML declares the names of all observable columns and one positive price
column. Keep forward targets and outcome-derived features in the label input.
For precomputed observation values, the declared availability must include any
upstream processing time. Null and NaN are missing. Infinity is rejected.
`price_return` compares positive same-symbol prices from consecutive supplied
decisions only when the latest visible event advanced. An unmatched, null,
nonpositive, or repeated-event price makes the affected return missing.
`trailing_volatility` is sample standard deviation over the configured number
of consecutive nonmissing causal price returns. Missing values break the window.
Supplied decision keys define the calendar; give every scheduled decision a row
to make an observation gap explicit.

## Configuration

The YAML requires `schema_version: 1` and the following sections. Unknown or
duplicate keys are rejected.

| Section | Fields |
| --- | --- |
| `inputs` | `observations`, `decisions`, `observation_columns`, `price_column` |
| `windows` | `train_seconds`, `validation_seconds`, `oos_seconds`, `step_seconds`, `embargo_seconds`, `mode` |
| `candidates` | Ordered list of unique `name`, `wrapper`, `features`, `missing`, and positive `ridge_penalty` for ridge |
| `metrics` | `periods_per_year`, `cost_bps`, `volatility_lookback`, `low_quantile`, `high_quantile` |
| `selection` | `rule` (`validation_sharpe`), `bootstrap_samples`, `block_length`, `min_holdout_periods`, `alpha`, `seed` |
| `output_directory` | Report directory |

Market volatility at each decision is the median of observable per-symbol
trailing volatilities. The workflow fits two linear-interpolated quantiles to
the eligible training cross-sections in each fold. Values at or below the lower
threshold are low, values above it and at or below the upper threshold are mid,
and the rest are high. A missing value or fewer than two qualifying training
dates yields an unclassified regime.

## Windows and wrappers

The earliest decision anchors the first fold. Adjacent train, validation and
OOS windows use wall-clock durations in UTC and half-open `[start, end)`
boundaries. Each step shifts validation and OOS by `step_seconds`. Rolling
training shifts by the same step. Expanding training keeps the original start.
The step must be at least the OOS duration, giving each OOS decision key a
single owner. Only windows whose OOS end is at or before the last supplied
decision timestamp are planned. A planned fold with an empty eligible train,
validation, or OOS role fails with its counts; a short panel with no complete
fold also fails.

For fitting, the protected cutoff is the validation start. For selection, it
is the first OOS start. For each role, apply these filters in order:

1. Embargo decisions at or after `cutoff - embargo` within the prior window.
2. Purge labels ending at or after the protected cutoff.
3. Require availability at or before the cutoff and a known label end.
4. Require a finite realized target.

The report counts removals once under the first applicable reason. A timestamp
determines the window for all its symbols; label eligibility can vary by symbol.
Ex-post OOS scoring uses realized labels with a known availability timestamp.
OOS outcomes can become eligible for later fold fits once they have matured.

The baseline copies one observable feature as the score. Ridge fits a linear
map with an unpenalized intercept and positive L2 penalty. Its columns use
training means and population standard deviations, with scale one for a
constant feature. Ridge requires at least two usable training rows. The
`drop` policy skips any incomplete feature vector. The `train_median` policy
fills partially missing vectors from eligible training medians and fails the
fit when a feature has no observed training value. Both skip an entirely missing
vector while preserving its key and null prediction. Validation and OOS
predictions reuse the fitted parameters for their fold. Neither wrapper uses
a random fit, forward fill, or outcome-based feature selection.

## Metrics and selection

`evaluate_signals` left-joins unique prediction and outcome keys. Every
decision timestamp belongs to one fold and one regime. Pearson IC and Spearman
rank IC use finite pairs in each cross-section. Ties receive their average rank.
Fewer than two pairs or a constant side makes the cross-section undefined.
Aggregate IC and rank IC are means of defined decision-level values with equal
weight per decision. ICIR divides mean IC by sample standard deviation of IC,
without annualization.

The hypothetical portfolio allocates +0.5 gross to symbols above the score
median and −0.5 below it, with equal weight per side. Median ties receive zero
weight. Without both sides the portfolio holds cash. At each decision it uses
only scores to set weights, with realized simple returns from that decision's
forward period applied afterward. Turnover is half the L1 distance between
consecutive target weights, including entries and exits from the previous
decision, starting from cash. Net return subtracts `cost_bps / 10000` per unit
of turnover. A decision with no finite score or a missing weighted return
makes its return unavailable. The workflow retains missing periods, and Sharpe,
geometric return and drawdown become undefined for a return series with gaps.

Annualized return compounds the net series and uses the configured number of
periods per year. Annualized Sharpe is mean net return over its sample standard
deviation times the square root of periods per year, using a zero risk-free
rate. Drawdown is the largest positive peak-to-trough loss, starting at equity
one. A loss of at least 100% exhausts the equity path. Fold and regime returns
are chronological subsequences with turnover inherited from the full sequence.
Combined metrics use the full ordered OOS series, including fold transitions.
For portfolio evaluation, realized returns must represent successive,
nonoverlapping rebalancing periods. If a weighted label ends after the next
evaluation decision, the portfolio marks its return undefined while retaining
IC and rank IC. The example is synthetic and excludes liquidity, market impact,
borrowing and financing costs.

The ordered candidate list is the trial ledger. Comparable candidates are
ranked by finite first-fold validation Sharpe, with configuration order breaking
ties. Within eligible validation periods, weights use the full decision
cross-section and the portfolio requires eligible realizations for its weights.
The winning configuration freezes at the start of the first OOS window.
Later folds refit that configuration at the declared cutoff. Trial IDs hash the
wrapper, ordered feature list, missing-data policy and penalty. Repeated designs
record a duplicate and refer to the first attempt. The ledger includes attempts
with insufficient training or undefined validation Sharpe.

The post-selection check tests a one-sided null of nonpositive Sharpe. It
centers the ordered OOS returns and draws seeded circular blocks of consecutive
returns with replacement, truncating each resample to the original length.
Its nominal p-value is `(1 + number of bootstrap Sharpes at least as large as observed) /
(1 + bootstrap_samples)`. Bonferroni adjustment multiplies that value by all
configured attempts, including failed and duplicate entries, then caps it at one.
This accounting is conservative across correlated candidates. The bootstrap
requires a complete return series without omitted decision dates inside the
holdout, a defined Sharpe, and at least the larger
of the configured minimum or four blocks. A failed check gives no evidence of
positive Sharpe at the declared threshold. An undefined check reports its reason.
The report includes the smallest resolvable adjusted p-value.

This is an approximate finite-sample check. It assumes representative,
weakly stationary returns over a regularly sampled research period index and a
block length that captures material serial dependence. It does not correct
for unrecorded experiments or regimes that violate that approximation. The
regression suite includes seeded AR(1) returns with zero drift and planted
positive drift, plus the end-to-end future-price perturbation across folds.

## Python API and evidence

```python
from quantcheck.pit_alpha import PurgedWalkForwardSplit, asof_align, evaluate_signals
```

- `asof_align(observations, decisions, value_columns=[...])` takes Polars
  frames or Parquet paths and returns a sorted Polars frame of decision keys,
  matched observation timestamps and requested numeric values.
- `PurgedWalkForwardSplit(train, validation, oos, step, embargo=..., mode=...)`
  takes `datetime.timedelta` durations. Its `split(label_frame)` yields folds
  with windows, eligibility counts and the three keyed partitions.
- `evaluate_signals(predictions, outcomes, periods_per_year=252, cost_bps=0)`
  takes Polars frames. Predictions contain `symbol`, `decision_ts`, positive
  integer `fold`, numeric `prediction`, and optional `regime`. Outcomes use
  the same key and `target_return`. Optional `label_end_ts` enables the overlap
  check. The result exposes period rows and
  combined, fold, regime, and fold-by-regime statistics.

JSON schema version 1 records the source SHA-256 digests, normalized research
configuration, fit and eligibility manifests, trial identities and ranks,
selection windows, bootstrap settings and results, and the chronological
metrics. Every statistic has a value or an undefined reason and a sample count.
The Markdown summary is rendered from the same result. `pit-alpha` uses
floating-point research statistics. The historical audit uses its existing
Decimal and canonical-string artifact contract.

See [the v1.0 release notes](RELEASE_NOTES_1.0.md) for qualification commands
and candidate evidence.
