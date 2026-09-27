#!/usr/bin/env python3
"""Prepare paired Spot and target-oracle prices from frozen Ethereum observations.

This does not run the AMM or select market parameters. Prices are WAD integers;
Spot is crvUSD/LP and the three-leg target oracle is USD/LP. The oracle and its
price_w schedule are reconstructed assumptions, not a historical deployment.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from simulator.pairs.reusd_sfrxusd_lp.collect_history import read_history, write_history

WAD = 10**18
A_PRECISION = 10**4


@dataclass(frozen=True)
class Config:
    ema_seconds: int = 866
    oracle_update_seconds: int = 0  # price_w at every recorded block
    initial_vp_scale: float = 1.0


def _x_from_y(a_raw, y):
    b = WAD - 4 * a_raw * (WAD - y) // A_PRECISION
    radicand = b * b + 4 * a_raw * WAD**3 // (A_PRECISION * y)
    return (math.isqrt(radicand) - b) * A_PRECISION // (8 * a_raw)


def _p_from_y(a_raw, y):
    x = _x_from_y(a_raw, y)
    a = 4 * a_raw * x // A_PRECISION
    return (a + WAD**3 // (4 * y * y)) * WAD // (a + WAD**3 // (4 * x * y))


@lru_cache(maxsize=65536)
def portfolio_value(a_raw, price):
    """Integer port of pinned curve-std lp_oracle_2 (coin zero).

    The pool's A_precise must be rescaled before calling this function.
    Keep rounding and the 1e-6 bisection price tolerance identical to Vyper.
    """
    if not (0 < a_raw <= 100000 * A_PRECISION and 10**16 <= price <= 10**20):
        raise ValueError("outside the contract's LP solver domain")
    target = (WAD * WAD + price // 2) // price if price < WAD else price
    lo, hi = 1, WAD // 2 + 1
    for _ in range(60):
        y = (lo + hi) // 2
        candidate = _p_from_y(a_raw, y)
        if abs(candidate - target) <= target // 10**6:
            break
        if candidate > target:
            lo = y
        else:
            hi = y
        if hi - lo <= 1:
            y = hi
            break
    else:
        raise ArithmeticError("LP solver did not converge")
    x = _x_from_y(a_raw, y)
    if price < WAD:
        x, y = y, x
    return x + price * y // WAD


@lru_cache(maxsize=8192)
def decay(dt, horizon):
    # Float exp is the only approximation to the contract's integer EMA math.
    return int(math.exp(-dt / horizon) * WAD)


class VirtualPriceEMA:
    def __init__(self, value, timestamp, horizon):
        self.previous = self.queued = value
        self.timestamp, self.horizon = timestamp, horizon

    def price(self, spot, timestamp, write=True):
        if timestamp < self.timestamp or self.horizon <= 0:
            raise ValueError("invalid EMA time")
        mul = decay(timestamp - self.timestamp, self.horizon)
        smoothed = (self.previous * mul + self.queued * (WAD - mul)) // WAD
        value = min(spot, smoothed)
        if write:
            self.previous = value
            self.queued = spot
            self.timestamp = timestamp
        return value


def feed_value(row):
    fee = row["base_redemption_fee"]
    if not 0 <= fee <= WAD or row["bridge_price_oracle"] <= 0:
        raise ValueError("invalid redemption inputs")
    modeled = max(WAD * WAD // row["bridge_price_oracle"], WAD - fee)
    observed = row["reusd_feed"]
    if observed is not None and observed != modeled:
        raise ValueError("observed feed disagrees with proposed reconstruction")
    return min(WAD, modeled if observed is None else observed)


def reconstruct(rows, config):
    if not rows:
        raise ValueError("empty history")
    ema = VirtualPriceEMA(
        int(Decimal(rows[0]["lp_virtual_price"]) * Decimal(str(config.initial_vp_scale))),
        rows[0]["timestamp"],
        config.ema_seconds,
    )
    points = []
    previous_block, previous_time = -1, -1
    for row in rows:
        timestamp, block = row["timestamp"], row["block"]
        if block <= previous_block or timestamp <= previous_time:
            raise ValueError("history must be strictly chronological")
        previous_block, previous_time = block, timestamp
        a_raw = row["lp_A_precise"] * A_PRECISION // 200
        vp = row["lp_virtual_price"]
        write = timestamp - ema.timestamp >= config.oracle_update_seconds
        smoothed = ema.price(vp, timestamp, write)
        lp_spot = portfolio_value(a_raw, row["lp_last_price"]) * vp // WAD
        lp_oracle = portfolio_value(a_raw, row["lp_price_oracle"]) * smoothed // WAD
        spot = lp_spot * (WAD * WAD // row["bridge_last_price"]) // WAD
        oracle = lp_oracle * feed_value(row) // WAD
        # Match ChainOracle's third leg and stepwise integer rounding. LLAMMA
        # consumes this USD-valued number directly; market trades remain crvUSD.
        aggregator = row["crvusd_agg_price"]
        if aggregator <= 0:
            raise ValueError("nonpositive USD aggregator price")
        oracle = oracle * aggregator // WAD
        if spot <= 0 or oracle <= 0:
            raise ValueError("nonpositive price")
        points.append((timestamp, block, spot, oracle))
    return points


def prepare(history: Path, output: Path):
    if output.exists():
        raise FileExistsError("Output already exists; use a new path to preserve frozen data")
    metadata, rows = read_history(history)
    if metadata.get("schema") != 4 or metadata.get("chain_id") != 1:
        raise ValueError("Expected schema-4 Ethereum history including the USD aggregator")
    config = Config()
    points = reconstruct(rows, config)
    header = {
        "schema": "reusd-sfrxusd-lp-prices-v1",
        "record_count": len(points),
        "history": metadata,
        "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("collect_history.py"), ROOT / "uv.lock")
        },
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "config": asdict(config),
        "target_source": "https://github.com/wavey0x/curve-stablecoin/tree/67b3b3b4057bf5d5f99b128b4b49cc9e2eae5f1a",
        "scale": WAD,
        "units": {"spot": "crvUSD per LP", "oracle": "USD per LP", "timestamp": "Unix seconds"},
        "oracle_state": "Initialized from the first LP virtual price; continuous price_w at each recorded block",
        "quote_convention": "Keep the USD aggregator in Oracle; Spot is independently valued in crvUSD. No alignment conversion.",
        "limitations": [
            "Point snapshots, not OHLC; denser sampling in recorded stress ranges",
            "The target oracle and its write schedule are counterfactual",
            "EMA exponentiation uses floating point; other price arithmetic uses integers",
            "All observations retained; loan eligibility and warm-up exclusions belong to the experiment",
        ],
    }
    records = ({"timestamp": t, "block": b, "spot": spot, "oracle": oracle} for t, b, spot, oracle in points)
    return write_history(output, header, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"sha256": prepare(args.history, args.output)}, sort_keys=True))
