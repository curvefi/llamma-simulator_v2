# reUSD/sfrxUSD LP–crvUSD parameter screen

This directory applies Curve's unchanged Python `LendingAMM` to the proposed
Ethereum reUSD/sfrxUSD LP market. The deployment script and this study both use
a 0.20% LLAMMA fee, but the simulation does **not** establish 0.20% as the
optimal fee.

## Result

| Parameter | Result | Interpretation |
| --- | ---: | --- |
| `A` at a fixed 0.20% fee | **210** | Minimum of the tested `A` sweep; the dense neighboring check gives 2.2210% required liquidation discount. |
| LLAMMA fee | **Unresolved; 0.20% provisional** | The loss-only sweep improves through 0.50%, its upper boundary. The model does not measure whether that fee delays arbitrage or increases later bad-debt risk. |
| Liquidation discount at `A=210`, fee 0.20% | **2.50% conditional recommendation** | Above the 2.2210% historical result and the 2.3415% external-execution-fee sensitivity envelope. |
| Loan discount, caps, and oracle EMA | **Not determined** | Outside this normalized soft-liquidation-loss model. |

The mechanical fee result and the economic recommendation are intentionally
separate. At `A=210`, the smallest modeled loss occurs at the highest fee
tested, 0.50%. A higher LLAMMA fee suppresses marginal arbitrage and earns more
fees in this replay, which can improve the loss metric. The same metric does
not penalize slower price tracking, time out of equilibrium, hard liquidation,
or final bad debt. Therefore 0.20% is retained only as a provisional economic
constraint. Selecting between 0.20% and 0.50% requires a separate study of
trade frequency, tracking error, conversion time, and bad debt.

The relevant external StableSwap pools charged 0.02% at the pinned block. A
0.50% LLAMMA base fee is 25 times that value; 0.20% is still 10 times it. This
comparison motivates additional review but does not by itself prove either fee
safe or unsafe.

## Method

The input is a pinned five-block Ethereum history of the three proposed price
legs:

```text
external mark = LP(last_price, virtual_price)
              × inverse bridge last_price
              × crvUSD/USD

lending oracle = LP(price_oracle, asymmetric virtual-price EMA)
               × inverse bridge price_oracle
               × crvUSD/USD
```

The observations are aggregated into five-minute market candles and aligned
oracle values. Each one-day window starts with one normalized collateral unit
in four LLAMMA bands. The calculation follows Curve's documented search order:

1. sweep `A` at the provisional 0.20% fee;
2. sweep fee at the selected `A`;
3. check `A=205, 210, 215` and the deployment script's `A=285` on hourly starts;
4. vary the assumed external execution fee from 0% to 0.50%.

The primary objective is maximum historical loss after applying Curve's
four-band coefficient:

```text
required discount = 1 - (1 - maximum raw loss) × band coefficient(A)
```

This maximum-loss objective is more conservative than the average-of-worst-
samples objective used by the repository's older pair scripts. It is an
explicit study assumption, not a change to the `LendingAMM` implementation.

## Files

- `collect_history.py` reads the three oracle legs from an archive RPC and
  writes pinned, checksummed raw observations.
- `calculate.py` reconstructs the market/oracle series, imports Curve's
  `LendingAMM` unchanged, and writes the concise evidence file.
- `data/REUSD_SFRXUSD_LP/reusd-sfrxusd-lp-5block-onchain.jsonl.gz` is the raw
  input, pinned through Ethereum block `25,870,305`.
- `results/REUSD_SFRXUSD_LP/summary.json` contains the complete reviewable
  sweeps and conclusions used above.

## Reproduce

```bash
python simulator/pairs/reusd_sfrxusd_lp/collect_history.py \
  --rpc-url "$ETH_RPC_URL" \
  --output data/REUSD_SFRXUSD_LP/reusd-sfrxusd-lp-5block-onchain.jsonl.gz \
  --end-block 25870305

python simulator/pairs/reusd_sfrxusd_lp/calculate.py \
  --history data/REUSD_SFRXUSD_LP/reusd-sfrxusd-lp-5block-onchain.jsonl.gz \
  --output results/REUSD_SFRXUSD_LP/summary.json \
  --processes 8 --seed 20260830 --random-windows 500
```

## Scope

The model measures normalized LLAMMA soft-liquidation loss. It does not model
size-aware LP exits, arbitrage capital, gas or MEV, pool-liquidity withdrawal,
hard liquidation, final bad debt, issuer/redemption/bridge failure, loan LTV,
caps, monetary policy, or the safety of the proposed 866-second oracle EMA.
Those questions must be resolved before activating nonzero exposure.
