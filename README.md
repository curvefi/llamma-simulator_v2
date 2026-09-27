# Curve LLAMMA Simulator

## Installation

### Locally using uv

```
pip install uv
uv venv
uv sync
```

### Recommended python version
It's recommended to use pypy to do faster simulations

```
uv python install pypy@3.11
uv venv --python pypy@3.11
uv sync -p pypy@3.11
```

## Running simulations

### Import data

Add pair to simulator/settings.py and import price data

```
python manage.py import_data {pair_name} (i.e. BTCUSDT)
```

### Performing calculations

_PDF will be added with more detailed explanation_

1) Change parameters in python manage.py for `calculate` command

2) Run simulations

```
python manage.py calculate
```
Results automatically will be saved in results folder.

### Separate scripts

Script ran for every pair is stored in `simulator/pairs` directory to save parameters used in calculations

```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/directory"
python simulator/pairs/btcusd/calculate_a.py
```

### reUSD/sfrxUSD LP market data

`data/REUSD_SFRXUSD_LP/history.jsonl.gz` contains 824,330 pinned Ethereum
observations from 2025-03-20 to 2026-08-30. `prices.jsonl.gz` in the same directory
contains the prepared Spot and target-oracle series. Both are gzip JSON Lines:
the first line contains `metadata`, followed by one record per observed block.
The prepared header records source/code hashes, configuration, units and runtime.
Each prepared record has `timestamp` (Unix seconds), `block`, `spot` and `oracle`;
prices are integers scaled by `10**18`; use arbitrary-precision integer parsing.
The historical pool A is an input
to LP valuation, independent of any LLAMMA A a later experiment chooses.

The target oracle is LP valuation using the pool EMA and dampened virtual price,
multiplied by the capped reUSD/crvUSD feed and then the crvUSD/USD aggregator.
It implements the [specified three-leg deployment](https://github.com/wavey0x/curve-stablecoin/blob/67b3b3b4057bf5d5f99b128b4b49cc9e2eae5f1a/scripts/deploy/llamalend/ethereum/markets/reUSDsfrxUSDLP-crvUSD/deploy.py).
Oracle is **USD/LP**; the independent Spot benchmark is **crvUSD/LP**, from pool
and bridge last prices. They are not silently converted into a common quote.
Integer scaling/rounding is preserved; EMA exponentiation uses floating point.
The LP oracle is counterfactual: state starts at the first observed virtual price,
uses an 866-second EMA and assumes `price_w` at every sampled block. All rows are
retained, including startup; window sampling and loan eligibility are separate
experiment choices. Observations are point snapshots, not OHLC, sampled every
five blocks with the header's denser stress ranges. This dataset does not include
simulation results or imply that upstream's OHLC loader accepts paired snapshots.

Rebuild prices offline with the locked environment (committed export: CPython
3.11.15). The output must be a new path; preparation never overwrites saved data:

```bash
uv sync --frozen --python 3.11.15
uv run --frozen python simulator/pairs/reusd_sfrxusd_lp/prepare.py \
  --history data/REUSD_SFRXUSD_LP/history.jsonl.gz --output .tmp/prices.jsonl.gz
cmp data/REUSD_SFRXUSD_LP/prices.jsonl.gz .tmp/prices.jsonl.gz
uv run --frozen python -m unittest discover -s tests -v
```

Gzip bytes can vary across compression runtimes; compare integer records and
source/configuration metadata when runtime versions differ. Optional compiled-contract
tests use `REUSD_ORACLE_SOURCE` pointing to the pinned `StableSwapNGLPOracle.vy`
and its locked contract environment; tests check source/dependency hashes.

Recollection is optional and requires `ETH_RPC_URL` in the environment:

```bash
uv run --frozen python simulator/pairs/reusd_sfrxusd_lp/collect_history.py \
  --end-block 25870305 \
  --dense-range 24340000:24380000 --dense-range 25740000:25785000 \
  --output .tmp/history.jsonl.gz
```

The history header records contracts, block hashes and any reused source history.
Fresh collection may change provenance metadata; compare the observations.
