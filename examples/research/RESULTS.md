# Weekly crypto momentum results

The first validation window selected 28-day momentum with a net Sharpe of
0.76 at 25 bps. It was the only candidate with a positive validation Sharpe.
The selected rule reached 0.43 over 182 held-out weeks from July 2022 through
December 2025. The circular block bootstrap returned a one-sided p-value of
0.183 before adjustment and 0.549 after adjustment for three trials. The
experiment did not meet its 0.05 threshold. The separate 95% circular-block
percentile interval for mean weekly net return was -0.24% to 0.70%. The interval
is descriptive and does not adjust for candidate selection.

| Fixed rule | Rank IC | Sharpe, 0 bps | Sharpe, 25 bps | Sharpe, 100 bps |
| --- | ---: | ---: | ---: | ---: |
| 28-day momentum | 0.045 | 0.56 | 0.43 | 0.05 |
| 28-day reversal | -0.045 | -0.56 | -0.68 | -1.06 |
| 7-day momentum | 0.021 | 0.91 | 0.66 | -0.09 |

The chosen rule returned 7.67% annualized after its primary cost, with a 20.88%
maximum drawdown and 23.24% mean weekly half-L1 turnover. At 100 bps its
annualized return fell to -1.65%. The 7-day rule led the holdout table at
the primary cost. Its validation Sharpe was -1.75, so it stayed outside the
primary result. The sensitivity rows are descriptive and use the same holdout
dates. Cash earned 0% by definition in this comparison.

Two runs using the pinned data snapshot and locked Python dependencies
produced identical report files and source hashes. [Study protocol and rerun command](README.md) ·
[Input URLs and SHA-256 hashes](SOURCES.json)

Market prices are from Coin Metrics under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The table contains research statistics calculated from these data.
