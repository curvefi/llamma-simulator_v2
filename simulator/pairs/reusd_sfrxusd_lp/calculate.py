#!/usr/bin/env python3
"""Reconstruct the proposed oracle and screen LLAMMA A and fee.

The Curve ``LendingAMM`` implementation is imported unchanged.  This file only
adapts the proposed reUSD/sfrxUSD LP market's composed price history to that
engine and records the small set of results used in the recommendation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


WAD = 10**18
A_PRECISION = 10**4
POOL_A_PRECISION = 100
N_COINS = 2
LP_VIRTUAL_PRICE_EMA_TIME = 866
CANDLE_SECONDS = 300
ONE_DAY_CANDLES = 86_400 // CANDLE_SECONDS
BANDS = 4
DYNAMIC_FEE_MULTIPLIER = 0.25
BASE_EXTERNAL_FEE = 0.0005
PROVISIONAL_BASE_FEE = 0.002

# Curve's documented process searches A at a fixed fee, then searches fee at
# the selected A.  The denser stable-asset range includes the script's A=285.
A_VALUES = (*range(60, 141, 10), *range(145, 231, 5), 250, 285, 325)
FEE_VALUES = (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005)
DENSE_A_VALUES = (205, 210, 215, 285)
EXTERNAL_FEE_VALUES = (0.0, 0.0005, 0.001, 0.002, 0.005)


@dataclass
class Series:
    candles: list[list[float]]
    oracle: list[float]
    stats: dict


MARKET: list[list[float]] = []
ORACLE: list[float] = []


def _x_from_y(a_raw: int, y: int) -> int:
    b1 = WAD - 4 * a_raw * (WAD - y) // A_PRECISION
    term = 4 * a_raw * WAD**3 // (A_PRECISION * y)
    return (math.isqrt(abs(b1) ** 2 + term) - b1) * A_PRECISION // (8 * a_raw)


def _p_from_y(a_raw: int, y: int) -> int:
    x = _x_from_y(a_raw, y)
    term_4a = 4 * a_raw * x // A_PRECISION
    return (term_4a + WAD**3 // (4 * y * y)) * WAD // (term_4a + WAD**3 // (4 * x * y))


def _y_from_price(a_raw: int, price: int) -> int:
    lo, hi = 1, WAD // 2 + 1
    for _ in range(60):
        mid = (lo + hi) // 2
        candidate = _p_from_y(a_raw, mid)
        if abs(candidate - price) <= price // 10**6:
            return mid
        if candidate > price:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1:
            return hi
    raise AssertionError("portfolio-value bisection did not converge")


def portfolio_value(a_raw: int, price: int) -> int:
    """Integer port of ``curve_std.stableswap.lp_oracle_2``."""
    if price < WAD:
        inverse = (WAD * WAD + price // 2) // price
        y_inverse = _y_from_price(a_raw, inverse)
        x_inverse = _x_from_y(a_raw, y_inverse)
        x, y = y_inverse, x_inverse
    else:
        y = _y_from_price(a_raw, price)
        x = _x_from_y(a_raw, y)
    return x + price * y // WAD


def read_history(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        first = json.loads(next(stream))
        if set(first) != {"metadata"}:
            raise ValueError("history has no metadata header")
        metadata = first["metadata"]
        observations = [json.loads(line) for line in stream]
    if len(observations) != metadata["record_count"]:
        raise ValueError("history record count does not match metadata")
    return metadata, observations


def reconstruct(observations: list[dict]) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    ema = observations[0]["lp_virtual_price"]
    queued = ema
    ema_timestamp = observations[0]["timestamp"]

    for observation in observations:
        timestamp = observation["timestamp"]
        virtual_price = observation["lp_virtual_price"]
        multiplier = math.exp(-max(timestamp - ema_timestamp, 0) / LP_VIRTUAL_PRICE_EMA_TIME)
        smoothed = int(ema * multiplier + queued * (1 - multiplier))
        if virtual_price < smoothed:
            oracle_virtual_price = virtual_price
            ema = queued = virtual_price
        else:
            oracle_virtual_price = smoothed
            ema, queued = smoothed, virtual_price
        ema_timestamp = timestamp

        a_raw = observation["lp_A_precise"] * A_PRECISION // (N_COINS ** (N_COINS - 1) * POOL_A_PRECISION)
        lp_spot = portfolio_value(a_raw, observation["lp_last_price"]) * virtual_price // WAD
        lp_oracle = portfolio_value(a_raw, observation["lp_price_oracle"]) * oracle_virtual_price // WAD
        bridge_spot = WAD * WAD // observation["bridge_last_price"]
        bridge_oracle = WAD * WAD // observation["bridge_price_oracle"]
        aggregate = observation["crvusd_agg_price"]
        spot = lp_spot * bridge_spot // WAD * aggregate // WAD
        oracle = lp_oracle * bridge_oracle // WAD * aggregate // WAD
        points.append((timestamp, spot / WAD, oracle / WAD))
    return points


def five_minute_series(points: list[tuple[int, float, float]]) -> Series:
    buckets: dict[int, list[tuple[int, float, float]]] = {}
    for point in points:
        buckets.setdefault(point[0] // CANDLE_SECONDS * CANDLE_SECONDS, []).append(point)

    candles: list[list[float]] = []
    oracle: list[float] = []
    for timestamp in sorted(buckets):
        rows = buckets[timestamp]
        spots = [row[1] for row in rows]
        candles.append([timestamp, spots[0], max(spots), min(spots), spots[-1], 0.0])
        oracle.append(rows[-1][2])

    gaps = [candles[index][0] - candles[index - 1][0] for index in range(1, len(candles))]
    spot_oracle_gaps = [abs(candle[4] / price - 1) for candle, price in zip(candles, oracle)]
    stats = {
        "observations": len(points),
        "candles": len(candles),
        "first_timestamp": int(candles[0][0]),
        "last_timestamp": int(candles[-1][0]),
        "max_candle_gap_seconds": max(gaps, default=0),
        "spot_min": min(candle[3] for candle in candles),
        "spot_max": max(candle[2] for candle in candles),
        "oracle_min": min(oracle),
        "oracle_max": max(oracle),
        "max_spot_oracle_gap": max(spot_oracle_gaps),
    }
    return Series(candles, oracle, stats)


def initialise_amm(A: int, fee: float, oracle: float, timestamp: float):
    from simulator.amm.lending_amm import LendingAMM

    p_base = oracle * (A / (A - 1) + 1e-4)
    amm = LendingAMM(p_base, A, fee, DYNAMIC_FEE_MULTIPLIER)
    amm.deposit_nrange(1.0, oracle, BANDS)
    initial_value = amm.get_all_x()

    # A fresh loan starts with oracle memory at its current oracle price.
    amm.p_oracle = oracle
    amm.prev_p_oracle = oracle
    amm.raw_p_oracle = oracle
    amm.old_p_oracle = oracle
    amm.old_dfee = 0.0
    amm.prev_p_oracle_time = timestamp
    amm.current_timestamp = timestamp
    return amm, initial_value


def simulate(prices: list[list[float]], oracles: list[float], A: int, fee: float, external_fee: float) -> float:
    if not prices or len(prices) != len(oracles):
        raise ValueError("market and oracle paths must be non-empty and aligned")
    amm, initial_value = initialise_amm(A, fee, oracles[0], prices[0][0])

    def target(external_price: float, timestamp: int, up: bool) -> float:
        bands = range(amm.max_band, amm.min_band - 1, -1) if up else range(amm.min_band, amm.max_band + 1)
        for band in bands:
            dynamic_fee = amm.dynamic_fee(band, timestamp=timestamp)
            boundary = amm.p_down(band) * (1 + dynamic_fee) if up else amm.p_up(band) * (1 - dynamic_fee)
            if (up and external_price > boundary) or (not up and external_price < boundary):
                return external_price * (1 - dynamic_fee if up else 1 + dynamic_fee)
        band = amm.min_band if up else amm.max_band
        dynamic_fee = amm.dynamic_fee(band, timestamp=timestamp)
        return external_price * (1 - dynamic_fee if up else 1 + dynamic_fee)

    for candle, oracle in zip(prices, oracles):
        timestamp, _open, high, low, _close, _volume = candle
        amm.set_p_oracle(oracle, timestamp=timestamp)
        for up, external_price in ((True, high), (False, low)):
            external_price *= 1 - external_fee if up else 1 + external_fee
            destination = target(external_price, int(timestamp), up)
            if up and destination > amm.get_p():
                amm.trade_to_price(destination)
            elif not up and destination < amm.get_p():
                amm.trade_to_price(destination)

    loss = 1 - amm.get_all_x() / initial_value
    if not math.isfinite(loss):
        raise ArithmeticError("simulation returned a non-finite loss")
    return loss


def simulate_task(task: tuple[int, float, int, float]) -> float:
    A, fee, start, external_fee = task
    end = start + ONE_DAY_CANDLES
    return simulate(MARKET[start:end], ORACLE[start:end], A, fee, external_fee)


def band_coefficient(A: int) -> float:
    return sum(((A - 1) / A) ** (band + 0.5) for band in range(BANDS)) / BANDS


def required_discount(A: int, raw_loss: float) -> float:
    return 1 - (1 - raw_loss) * band_coefficient(A)


def evaluate(pool, A: int, fee: float, starts: Iterable[int], external_fee: float = BASE_EXTERNAL_FEE) -> dict:
    starts = list(starts)
    tasks = ((A, fee, start, external_fee) for start in starts)
    losses = list(pool.imap(simulate_task, tasks, chunksize=16))
    if not losses or len(losses) != len(starts) or any(not math.isfinite(loss) for loss in losses):
        raise RuntimeError("simulation did not return every requested window")
    maximum = max(losses)
    maximum_index = losses.index(maximum)
    return {
        "A": A,
        "fee": fee,
        "external_fee": external_fee,
        "windows": len(starts),
        "raw_loss_max": maximum,
        "required_liquidation_discount": required_discount(A, maximum),
        "max_window_start_timestamp": int(MARKET[starts[maximum_index]][0]),
    }


def separated_top_indices(values: list[float], count: int, separation: int) -> list[int]:
    selected: list[int] = []
    for index in sorted(range(len(values)), key=values.__getitem__, reverse=True):
        if all(abs(index - existing) >= separation for existing in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def calibration_starts(candles: list[list[float]], oracle: list[float], seed: int, random_count: int) -> list[int]:
    population = len(candles) - ONE_DAY_CANDLES
    starts = set(range(0, population, ONE_DAY_CANDLES))
    starts.update(random.Random(seed).sample(range(population), min(random_count, population)))

    rolling_ranges = []
    for start in range(population):
        window = candles[start : start + ONE_DAY_CANDLES]
        high = max(row[2] for row in window)
        low = min(row[3] for row in window)
        rolling_ranges.append((high - low) / high)
    gaps = [abs(row[4] / price - 1) for row, price in zip(candles[:population], oracle)]
    anchors = separated_top_indices(rolling_ranges, 5, ONE_DAY_CANDLES)
    anchors += separated_top_indices(gaps, 5, ONE_DAY_CANDLES)
    for anchor in anchors:
        starts.update(
            range(
                max(0, anchor - ONE_DAY_CANDLES),
                min(population, anchor + ONE_DAY_CANDLES + 1),
                12,
            )
        )
    return sorted(starts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--random-windows", type=int, default=500)
    args = parser.parse_args()
    if args.processes <= 0 or args.random_windows < 0:
        raise SystemExit("processes must be positive and random-windows cannot be negative")

    history_metadata, observations = read_history(args.history)
    series = five_minute_series(reconstruct(observations))
    if series.stats["max_candle_gap_seconds"] > CANDLE_SECONDS:
        raise ValueError("history contains a gap larger than one simulation candle")

    global MARKET, ORACLE
    MARKET, ORACLE = series.candles, series.oracle
    calibration = calibration_starts(MARKET, ORACLE, args.seed, args.random_windows)
    dense_hourly = range(0, len(MARKET) - ONE_DAY_CANDLES + 1, 12)

    context = mp.get_context("fork")
    with context.Pool(args.processes) as pool:
        a_sweep = [evaluate(pool, A, PROVISIONAL_BASE_FEE, calibration) for A in A_VALUES]
        selected_a = min(a_sweep, key=lambda row: row["required_liquidation_discount"])["A"]
        fee_sweep = [evaluate(pool, selected_a, fee, calibration) for fee in FEE_VALUES]
        dense_check = [evaluate(pool, A, PROVISIONAL_BASE_FEE, dense_hourly) for A in DENSE_A_VALUES]
        external_fee_sensitivity = [
            evaluate(pool, selected_a, PROVISIONAL_BASE_FEE, calibration, external_fee)
            for external_fee in EXTERNAL_FEE_VALUES
        ]

    candidate = next(row for row in dense_check if row["A"] == selected_a)
    fee_minimum = min(fee_sweep, key=lambda row: row["required_liquidation_discount"])
    external_envelope = max(row["required_liquidation_discount"] for row in external_fee_sensitivity)
    output = {
        "schema": 1,
        "history": history_metadata,
        "history_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "series": series.stats,
        "method": {
            "engine": "unmodified simulator.amm.lending_amm.LendingAMM",
            "search_order": "A at fixed 0.20% fee, then fee at selected A",
            "candle_seconds": CANDLE_SECONDS,
            "loan_days": 1,
            "bands": BANDS,
            "dynamic_fee_multiplier": DYNAMIC_FEE_MULTIPLIER,
            "base_external_fee": BASE_EXTERNAL_FEE,
            "calibration_windows": len(calibration),
            "dense_hourly_windows": len(dense_hourly),
            "seed": args.seed,
        },
        "a_sweep_at_0_20_percent_fee": a_sweep,
        "fee_sweep_at_selected_a": fee_sweep,
        "dense_a_check_at_0_20_percent_fee": dense_check,
        "external_fee_sensitivity": external_fee_sensitivity,
        "conclusions": {
            "selected_a_at_0_20_percent_fee": selected_a,
            "candidate": candidate,
            "loss_only_fee_minimum": fee_minimum,
            "fee_status": (
                "0.20% is a provisional economic constraint, not the simulated optimum. "
                "The loss-only minimum is the 0.50% upper search boundary. This model does "
                "not measure arbitrage responsiveness, time out of equilibrium, hard "
                "liquidation, or final bad debt, so it cannot resolve the fee choice."
            ),
            "external_fee_adjusted_discount_envelope": external_envelope,
            "conditional_liquidation_discount": 0.025,
            "not_determined": [
                "loan discount",
                "borrow cap",
                "vault supply limit",
                "oracle EMA time",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["conclusions"], indent=2))


if __name__ == "__main__":
    main()
