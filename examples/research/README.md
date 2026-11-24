# Weekly crypto momentum

The hypothesis is that ranking ten USD crypto price marks by their trailing
28-day returns produces a positive net long-short Sharpe on unseen weeks. The
first validation window chooses among 28-day momentum, its reversal, and 7-day
momentum. Zero-return cash provides the reference for the net Sharpe test.
Standalone runs show each rule at 0, 25, and 100 bps per unit of half-L1
turnover, including initial entry. Use 25 bps as the primary cost assumption.

The panel uses Monday UTC closes from January 2020 through December 2025.
A close labeled Monday becomes Tuesday 00:00 UTC in the panel. Signals use
closes marked two days before each decision. Each label covers the
next seven days, and becomes available to the experiment two days after that.
The first 104 weeks are training, the following 26 are validation, and
subsequent 26-week evaluation blocks are disjoint. Candidate selection occurs
once. The bootstrap uses circular eight-week blocks with a family-size
adjustment for the three attempted candidates.

Run from the repository root with Python 3.12 and the locked environment.

```sh
uv sync --frozen
uv run python examples/research/coinmetrics_momentum.py /tmp/pit-alpha-momentum
```

This is a noncommercial educational study. Data come from the Coin Metrics
Community archive, pinned to revision `f1a36afb962731c387bb03982758ab0103063da5`.
The runner records source hashes and fails on missing required rate dates.
The raw rates and generated panels stay out of Git. The two-day availability
lag is an assumption. The archive snapshot does not expose historical release
or revision timestamps. The universe is a retrospective set of ten long-lived
assets, so selection and survivorship bias remain. Coin Metrics PriceUSD values are market
aggregates, and the experiment assumes trades at those marks. Costs exclude
borrow, funding, market impact and venue-specific constraints.
[Coin Metrics archive and data license](https://github.com/coinmetrics/data/blob/f1a36afb962731c387bb03982758ab0103063da5/README.md) ·
[Coin Metrics PriceUSD documentation](https://github.com/coinmetrics/docs-website/blob/master/asset-metrics/market/priceusd.md)
