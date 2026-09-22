# reUSD/sfrxUSD LP–crvUSD parameter screen

This screen reconstructs paired oracle and market prices in **crvUSD per LP**
and replays them through the unchanged `LendingAMM`. It produces historical
soft-liquidation-loss comparisons, not launch parameter recommendations.

## Reproduce offline

After `uv sync --frozen --python 3.11`, run from the repository root:

```bash
uv run --frozen python simulator/pairs/reusd_sfrxusd_lp/calculate.py \
  --history data/REUSD_SFRXUSD_LP/history.jsonl.gz \
  --output results/REUSD_SFRXUSD_LP/summary.json
```

The summary records the complete configuration, data/code checksums, shared
window starts, and every evaluated A/fee combination. `--workers 1` runs the
same calculation serially. The committed result uses CPython 3.11.15.

## Prices and provenance

The frozen input contains **824,330 paired end-of-block observations** from
20 March 2025 14:25:47 UTC through 30 August 2026 19:53:11 UTC. Its last block is
25,870,305, hash
`0x9667d83b47fc3a5dbffd2ad99db063cda226b43e7cd6423c139332ac0938f48b`.
Each row contains its block/hash/time, both pools' last prices and existing
price EMAs, LP virtual price and A, the historical redemption handler and fee,
and the directly observed reUSD feed when deployed. Addresses are in the header;
method selectors and scaling are explicit in `collect_history.py`.

Sampling is every five blocks, with **every block** included in ranges
24,340,000–24,380,000 and 25,740,000–25,785,000 around the February and August
stress periods. The largest time gap elsewhere is 156 seconds. End-of-block
reads do not capture transaction ordering within a block.

```text
LP(p, vp, A) = StableSwap portfolio value(p, scaled A) × vp

spot = LP(pool.last_price, pool.virtual_price, pool.A)
       × inverse bridge.last_price

reUSD feed = max(inverse bridge.price_oracle, 1 − baseRedemptionFee)
oracle = LP(pool.price_oracle, dampened virtual price, pool.A)
         × min(1, reUSD feed)
```

All products above omit the implementation's explicit 1e18 divisions. The pool
A conversion is `A_precise × 10000 / 200`. The LP mark includes yield through
the pool inputs. The bridge pool prices its yield-bearing coin in underlying
units; the inverse supplies crvUSD/reUSD. Both final series use the debt asset's
numeraire, so neither includes a crvUSD/USD aggregator.

The 866-second virtual-price EMA persists across the entire history. An upward
write queues the new value for subsequent smoothing; a downward write resets
it immediately. Read-only calls return `min(current VP, EMA)` without changing
state. The base replay calls `price_w` at each sampled block. This is a
counterfactual call schedule, not a reconstruction of actual market activity.
The first day is excluded from loan starts as warm-up.

The deployed feed was read and matched against its formula on all **787,237**
postdeployment observations. The earlier **37,093** rows have an explicit null
feed and reconstruct the proposed recipe using historical pool/fee inputs.
The LP oracle history is also a reconstruction of the proposed design. A
redemption floor can keep this oracle above the market during a discount; the
largest positive oracle/spot gap in the dataset is 3.2587%, at block 24,361,675.
The floor is a valuation rule, not a guarantee of immediate executable liquidity.

The collection enriched the original pinned pool history, rechecking deterministic
samples against the RPC. Its enrichment-cache SHA is recorded in the header's
`reused_history`; the original source is retained in Git at
`ea68dda40502a49a864a31074c05e1a96a15062b`. Block hashes, handler, fee and feed
were freshly read for every final observation. Missing deployed reads or formula
mismatches abort collection. To recollect all inputs without reuse:

```bash
# Set ETH_RPC_URL to an Ethereum archive endpoint in the environment.
uv run --frozen python simulator/pairs/reusd_sfrxusd_lp/collect_history.py \
  --end-block 25870305 \
  --dense-range 24340000:24380000 --dense-range 25740000:25785000 \
  --output data/REUSD_SFRXUSD_LP/history.jsonl.gz
```

Fresh collection has different provenance metadata; paired raw readings should
match. Its checksum therefore need not match the supplied enrichment artifact.

## Replay and results

Each loan starts with one collateral unit in four bands. Every observation sets
that timestamp's oracle before considering an arbitrage trade against that
same timestamp's spot, with a 5 bp external execution cost. There are no OHLC
extrema or future oracle observations in the replay. Oracle memory starts at
the current oracle; initial and final value use `get_all_x()`.

The screen uses 652 shared one-day windows: daily starts plus hourly starts
preceding five separated oracle/market dislocations. It tests A at a fixed
0.20% fee, refines between the neighboring coarse A samples, checks the local
five-unit neighbors, and then tests fees from 0.10% through 0.50%.

```text
band-adjusted loss = 1 − (1 − maximum raw loss)
                        × mean(((A − 1) / A) ** (band + 0.5))
```

Among these 38 evaluations, A=850 at fee 0.20% has band-adjusted loss
**0.5412%** (raw loss 0.3068%). At that A, the tested fee outcomes are:

| Fee | Band-adjusted loss |
| --- | ---: |
| 0.10% | 0.6153% |
| 0.20% | 0.5412% |
| 0.30% | 0.5725% |
| 0.40% | 0.5725% |
| 0.50% | 0.5725% |

These are finite-grid, finite-window results. Neither the lowest tested A/fee
combination nor the adjusted loss is a recommended market configuration or a
proved minimum discount. `--exact --a-values A --fees FEE` evaluates a supplied
combination without selecting neighbors. `--window-step-seconds 3600` checks
hourly starts; `--windows-from RESULT.json` holds starts fixed across scenarios.

The hourly-start check covers **12,755 windows**. At the same 0.20% fee, adjusted
losses for A=845/850/855 are **0.7863% / 0.7882% / 0.7900%**. This changes the
ordering of the neighbors and raises the measured maximum: the quick screen
does not establish the best A or a sufficient discount. Reproduce that check:

```bash
uv run --frozen python simulator/pairs/reusd_sfrxusd_lp/calculate.py \
  --history data/REUSD_SFRXUSD_LP/history.jsonl.gz --output .tmp/hourly.json \
  --exact --a-values 845,850,855 --fees .002 --window-step-seconds 3600
```

## Checks and limits

`uv run --frozen python -m unittest discover -s tests -v` checks floor/cap
behavior, debt-asset units, EMA state, chronological replay, sampling, flat-price
normalization, and agreement between serial and parallel execution.

Sensitivity checks hold fixed the 626 quick-screen starts whose blocks also
lie on the five-block grid. At A=850 / fee 0.20%, adjusted loss is 0.5412% with
sample-by-sample writes, 0.5415% with 300-second writes, and 0.5979% with
3,600-second writes. The maximum post-warm-up oracle-price difference from
the base path is 0.00534% for 3,600-second writes. Starting the VP state 1%
higher or lower leaves the post-warm-up floating-point prices and losses
unchanged. Pure five-block sampling gives the same maximum for this case on
these shared starts; that is not a bound on missed paths. The sampling check
also changes the assumed write frequency. External costs of 0/5/10/50 bp
give adjusted losses of 0.5755% / 0.5412% / 0.5485% / 0.5725% on these starts.
These are individual sensitivities, not a combined stress envelope.

The optional contract test compares against compiled
[StableSwapNGLPOracle at cf1d05f](https://github.com/curvefi/curve-stablecoin/blob/cf1d05fb6bf7c608973cc41786b2e1fd81dc3a6a/curve_stablecoin/price_oracles/v2/StableSwapNGLPOracle.vy),
with curve-std `048cb23d0ed4c48768815ea5c46d4a676da8de35` and stableswap-ng
`7a9f6f11fb67e4778fb2496e65a09ec2939e339c`. The test verifies these dependency
pins and the oracle source hash. Historical inputs and synthetic up/down/same-block
transitions are compared with writes at each observation, every 300 seconds,
and every 3,600 seconds. Across 5,424 comparisons, the maximum difference was
**7 wei**; only EMA exponentiation uses floating point. The LP integer solver
matched the compiled implementation exactly on its tested domain vectors.
With that checkout's locked test dependencies installed:

```bash
REUSD_ORACLE_SOURCE=/path/to/curve-stablecoin/curve_stablecoin/price_oracles/v2/StableSwapNGLPOracle.vy \
  /path/to/curve-stablecoin/.venv/bin/python -m unittest discover -s tests -v
```

The model excludes size-dependent LP exits, liquidity depth and withdrawal,
arbitrage capital, gas/MEV, hard liquidation, final bad debt, issuer/redemption
failure, loan LTV, caps, and monetary policy. Prices are pool marginal marks,
not executable quotes for a specified position size. Sampling, start times,
and the assumed oracle-write schedule remain model assumptions.
