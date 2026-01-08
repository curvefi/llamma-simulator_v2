"""
Async on-chain chart importer that mirrors the Binance importer output format.

Environment
-----------
- ONCHAIN_RPC_URL: HTTPS RPC endpoint consumed by AsyncWeb3 (default http://localhost:8545).
- ONCHAIN_CHAIN_ID: Numeric chain id used for context metadata (default 1).
- ONCHAIN_BLOCK_TIME_SECONDS: Average seconds per block for timestamp-to-block mapping (default 12).
- ONCHAIN_ANCHOR_BLOCK: Reference block number that lines up with ONCHAIN_ANCHOR_TIMESTAMP (default 0).
- ONCHAIN_ANCHOR_TIMESTAMP: Reference unix timestamp in seconds paired with ONCHAIN_ANCHOR_BLOCK (default 0).
- ONCHAIN_PRICE_DECIMALS: Fixed-point decimals of the price returned by `get_price_at_block` (default 18).
- ONCHAIN_SAMPLES_PER_INTERVAL: Number of evenly spaced block samples per candle (default 4, min 2).
- ONCHAIN_RPC_CONCURRENCY: Maximum simultaneous on-chain reads (default 8).

Timestamp to block mapping is linear and deterministic:
    block = anchor_block + (timestamp - anchor_timestamp) / block_time_seconds
Negative results clamp to anchor_block. This keeps candles reproducible with the placeholder logic while
remaining simple to override when a production mapping is provided.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from web3 import AsyncWeb3
from web3.providers.async_rpc import AsyncHTTPProvider

from simulator.settings import Pair

from .base import BaseImporter

logger = logging.getLogger(__name__)
UTC = dt.timezone.utc


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid value %s for %s; falling back to %s", value, name, default)
        return default


def _env_dt(var: str, default: dt.datetime) -> dt.datetime:
    value = os.getenv(var)
    if not value:
        return default
    try:
        ts = int(value)
        return dt.datetime.fromtimestamp(ts, tz=UTC)
    except ValueError:
        logger.warning("Expected unix timestamp in seconds for %s; using default %s", var, default)
        return default


@dataclass(frozen=True)
class OnchainChartConfig:
    rpc_url: str
    chain_id: int
    block_time_seconds: int
    anchor_block: int
    anchor_timestamp: int
    price_decimals: int
    samples_per_interval: int
    max_concurrency: int


@dataclass(frozen=True)
class OnchainPriceContext:
    decimals: int
    pair: str
    chain_id: int


def default_config() -> OnchainChartConfig:
    return OnchainChartConfig(
        rpc_url=os.getenv("ONCHAIN_RPC_URL", "http://localhost:8545"),
        chain_id=_get_env_int("ONCHAIN_CHAIN_ID", 1),
        block_time_seconds=max(1, _get_env_int("ONCHAIN_BLOCK_TIME_SECONDS", 12)),
        anchor_block=_get_env_int("ONCHAIN_ANCHOR_BLOCK", 0),
        anchor_timestamp=_get_env_int("ONCHAIN_ANCHOR_TIMESTAMP", 0),
        price_decimals=_get_env_int("ONCHAIN_PRICE_DECIMALS", 18),
        samples_per_interval=max(2, _get_env_int("ONCHAIN_SAMPLES_PER_INTERVAL", 4)),
        max_concurrency=max(1, _get_env_int("ONCHAIN_RPC_CONCURRENCY", 8)),
    )


async def get_price_at_block(w3: AsyncWeb3, block_number: int, *, ctx: OnchainPriceContext) -> int:
    """
    Placeholder hook that fetches data from chain and returns a fixed-point price.

    The default implementation proxies `eth.get_block` and folds the base fee and block number into
    a deterministic pseudo price. Replace this with an actual contract read when integrating a real oracle.
    """

    block = await w3.eth.get_block(block_number)
    base_fee = block.get("baseFeePerGas") or 0
    pseudo_price = base_fee + (block_number % 1_000_000_000)
    scale = 10**ctx.decimals
    if pseudo_price == 0:
        pseudo_price = 10**4  # ensure non-zero to keep downstream ratios defined
    return int(pseudo_price * scale // max(1, 10**9))


class OnchainChartImporter(BaseImporter):
    name = "onchain"
    interval = os.getenv("ONCHAIN_INTERVAL", "1m")
    start = _env_dt("ONCHAIN_START_TS", dt.datetime(2022, 1, 1, tzinfo=UTC))
    end = _env_dt("ONCHAIN_END_TS", dt.datetime.now(dt.timezone.utc))
    config: OnchainChartConfig = default_config()

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        unit = interval[-1]
        value = interval[:-1]
        unit_map = {"m": 60, "h": 3600, "d": 86400}
        if unit not in unit_map:
            raise ValueError(f"Unsupported interval unit {interval}")
        return int(value) * unit_map[unit]

    @staticmethod
    def _align_start(ts: int, interval_seconds: int) -> int:
        aligned = (ts // interval_seconds) * interval_seconds
        if aligned < ts:
            aligned += interval_seconds
        return aligned

    @staticmethod
    def _interval_starts(start_ts: int, end_ts: int, interval_seconds: int) -> Iterable[int]:
        cur = start_ts
        while cur < end_ts:
            yield cur
            cur += interval_seconds

    @staticmethod
    def _timestamp_to_block(ts: int, config: OnchainChartConfig) -> int:
        delta = ts - config.anchor_timestamp
        if delta <= 0:
            return config.anchor_block
        return config.anchor_block + delta // config.block_time_seconds

    @staticmethod
    def _sample_points(start_ts: int, end_ts: int, count: int) -> list[int]:
        if count <= 1:
            return [start_ts]
        duration = max(1, end_ts - start_ts)
        step = duration / (count - 1)
        return [int(start_ts + round(step * idx)) for idx in range(count)]

    @classmethod
    async def _fetch_price_for_timestamp(
        cls,
        sem: asyncio.Semaphore,
        w3: AsyncWeb3,
        timestamp: int,
        ctx: OnchainPriceContext,
        config: OnchainChartConfig,
    ) -> float:
        block_number = cls._timestamp_to_block(timestamp, config)
        async with sem:
            fixed_point_price = await get_price_at_block(w3, block_number, ctx=ctx)
        return fixed_point_price / (10**ctx.decimals)

    @classmethod
    async def _build_candle(
        cls,
        sem: asyncio.Semaphore,
        w3: AsyncWeb3,
        interval_start: int,
        interval_seconds: int,
        end_ts: int,
        ctx: OnchainPriceContext,
        config: OnchainChartConfig,
    ) -> list[Any]:
        interval_end = min(interval_start + interval_seconds, end_ts)
        timestamps = cls._sample_points(interval_start, interval_end, config.samples_per_interval)
        prices = await asyncio.gather(
            *[cls._fetch_price_for_timestamp(sem, w3, ts, ctx, config) for ts in timestamps]
        )
        open_price = prices[0]
        close_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)
        # Placeholder volumes match Binance schema (base volume, quote volume)
        return [interval_start, open_price, high_price, low_price, close_price, 0.0, 0.0]

    @classmethod
    async def get_klines(
        cls, symbol: str, interval: str, start_ts: int, end_ts: int, **kwargs: Any
    ) -> list[list[Any]]:
        if start_ts >= end_ts:
            return []

        config: OnchainChartConfig = kwargs.get("config") or cls.config
        w3: AsyncWeb3 | None = kwargs.get("web3")

        if w3 is None:
            w3 = AsyncWeb3(AsyncHTTPProvider(config.rpc_url))

        ctx: OnchainPriceContext = kwargs.get("price_context") or OnchainPriceContext(
            decimals=config.price_decimals, pair=str(symbol), chain_id=config.chain_id
        )

        interval_seconds = cls._interval_to_seconds(interval)
        start_aligned = cls._align_start(start_ts, interval_seconds)
        if start_aligned >= end_ts:
            return []
        sem = asyncio.Semaphore(config.max_concurrency)

        tasks = [
            cls._build_candle(sem, w3, window_start, interval_seconds, end_ts, ctx, config)
            for window_start in cls._interval_starts(start_aligned, end_ts, interval_seconds)
        ]
        candles = await asyncio.gather(*tasks)
        # Ensure deterministic ordering regardless of gather completion timing
        candles.sort(key=lambda row: row[0])
        return candles

    @classmethod
    async def fetch(cls, pair: Pair) -> list[Any]:
        start_ts = int(cls.start.timestamp())
        end_ts = int(cls.end.timestamp())
        return await cls.get_klines(pair, cls.interval, start_ts, end_ts)

    @classmethod
    def load(cls, pair: Pair) -> list[Any]:
        path = cls.get_data_path(pair)
        with gzip.open(path, "r") as f:
            return json.load(f)


def _is_binance_schema(row: Sequence[Any]) -> bool:
    return (
        isinstance(row, list)
        and len(row) == 7
        and isinstance(row[0], (int, float))
        and all(isinstance(value, (int, float)) for value in row[1:])
    )


async def _example() -> None:
    """
    Minimal async usage example that fetches 5 minutes of data and validates the schema.
    """

    now = int(dt.datetime.now(tz=UTC).timestamp())
    start_ts = now - 5 * 60
    klines = await OnchainChartImporter.get_klines("ETHUSDT", "1m", start_ts, now)
    assert all(_is_binance_schema(kline) for kline in klines), "Unexpected schema"
    logger.info("Fetched %s klines; first=%s", len(klines), klines[0] if klines else None)


if __name__ == "__main__":
    asyncio.run(_example())
