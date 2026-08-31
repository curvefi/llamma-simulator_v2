#!/usr/bin/env python3
"""Collect reproducible Ethereum history for the reUSD/sfrxUSD LP market.

The proposed lending oracle is a three-leg product:

    LP/reUSD * reUSD/crvUSD * crvUSD/USD

This collector stores the pool and aggregator inputs needed to reconstruct
both the external mark and that oracle at historical blocks.  It supports
extending a previously checksummed data set without refetching old blocks.
RPC URLs are runtime-only inputs and are never written to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from eth_abi import decode, encode
from eth_utils import keccak

CHAIN_ID = 1
# First five-block sample at which every required pool/oracle call succeeds.
START_BLOCK = 22_088_660
BLOCK_STEP = 5

LP_POOL = "0xed785af60bed688baa8990cd5c4166221599a441"
BRIDGE_POOL = "0xc522a6606bba746d7960404f22a3db936b6f4f50"
CRVUSD_AGG = "0x18672b1b0c623a30089a280ed9256379fb0e4e62"
MULTICALL3 = "0xca11bde05977b3631167028862be2a173976ca11"


def calldata(signature: str, types: tuple[str, ...] = (), args: tuple = ()) -> str:
    encoded = keccak(text=signature)[:4]
    if types:
        encoded += encode(types, args)
    return "0x" + encoded.hex()


CALLS = (
    ("timestamp", MULTICALL3, calldata("getCurrentBlockTimestamp()")),
    ("lp_A_precise", LP_POOL, calldata("A_precise()")),
    ("lp_last_price", LP_POOL, calldata("last_price(uint256)", ("uint256",), (0,))),
    ("lp_price_oracle", LP_POOL, calldata("price_oracle(uint256)", ("uint256",), (0,))),
    ("lp_virtual_price", LP_POOL, calldata("get_virtual_price()")),
    (
        "bridge_last_price",
        BRIDGE_POOL,
        calldata("last_price(uint256)", ("uint256",), (0,)),
    ),
    (
        "bridge_price_oracle",
        BRIDGE_POOL,
        calldata("price_oracle(uint256)", ("uint256",), (0,)),
    ),
    ("crvusd_agg_price", CRVUSD_AGG, calldata("price()")),
)

MULTICALL_DATA = calldata(
    "aggregate3((address,bool,bytes)[])",
    ("(address,bool,bytes)[]",),
    ([(target, True, bytes.fromhex(data[2:])) for _, target, data in CALLS],),
)


@dataclass(frozen=True)
class Config:
    rpc_url: str
    output: Path
    base_history: Path | None
    start_block: int
    end_block: int | None
    block_step: int
    samples_per_request: int
    concurrency: int
    request_delay: float


async def rpc(session: aiohttp.ClientSession, url: str, method: str, params: list) -> object:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with session.post(url, json=payload) as response:
        if response.status >= 400:
            raise RuntimeError(f"RPC {method} returned HTTP {response.status}")
        item = await response.json()
    if "error" in item:
        raise RuntimeError(f"RPC {method} failed: {item['error']}")
    return item["result"]


def rpc_batch(blocks: list[int]) -> list[dict]:
    return [
        {
            "jsonrpc": "2.0",
            "id": sample_index,
            "method": "eth_call",
            "params": [{"to": MULTICALL3, "data": MULTICALL_DATA}, hex(block)],
        }
        for sample_index, block in enumerate(blocks)
    ]


async def fetch_chunk(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    config: Config,
    blocks: list[int],
) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            async with semaphore:
                async with session.post(config.rpc_url, json=rpc_batch(blocks)) as response:
                    if response.status >= 400:
                        raise RuntimeError(f"RPC batch returned HTTP {response.status}")
                    result = await response.json()
                if config.request_delay:
                    await asyncio.sleep(config.request_delay)
            by_id = {item["id"]: item for item in result}
            records: list[dict] = []
            for sample_index, block in enumerate(blocks):
                record = {"block": block}
                item = by_id[sample_index]
                if "error" in item or item.get("result") in (None, "0x"):
                    continue
                values = decode(["(bool,bytes)[]"], bytes.fromhex(item["result"][2:]))[0]
                if len(values) != len(CALLS) or not all(success for success, _ in values):
                    continue
                for (field, _, _), (_, value) in zip(CALLS, values):
                    if len(value) != 32:
                        raise ValueError(f"unexpected returndata length for {field}")
                    record[field] = int.from_bytes(value)
                records.append(record)
            if len(records) != len(blocks):
                raise RuntimeError("RPC batch returned only a partial chunk")
            return records
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt == 5:
                break
            await asyncio.sleep(0.5 * 2**attempt)
    assert last_error is not None
    raise RuntimeError("RPC batch failed after six attempts") from None


def read_history(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        first = json.loads(next(stream))
        if set(first) != {"metadata"}:
            raise ValueError(f"{path} has no metadata header")
        metadata = first["metadata"]
        records = [json.loads(line) for line in stream]
    if len(records) != metadata["record_count"]:
        raise ValueError(f"{path} record count does not match its metadata")
    return metadata, records


def validate_base(metadata: dict, records: list[dict], config: Config) -> None:
    if metadata["chain_id"] != CHAIN_ID:
        raise ValueError("base history is not Ethereum mainnet")
    if metadata["block_step"] != config.block_step:
        raise ValueError("base history uses a different block step")
    expected = {
        "lp_pool": LP_POOL,
        "bridge_pool": BRIDGE_POOL,
        "crvusd_aggregator": CRVUSD_AGG,
    }
    actual = {key: value.lower() for key, value in metadata["contracts"].items()}
    if actual != expected:
        raise ValueError("base history uses different contracts")
    if records:
        for previous, current in zip(records, records[1:]):
            if current["block"] <= previous["block"]:
                raise ValueError("base history is not strictly block ordered")


async def collect(config: Config) -> tuple[list[dict], dict]:
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=config.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        chain_id = int(await rpc(session, config.rpc_url, "eth_chainId", []), 16)
        if chain_id != CHAIN_ID:
            raise ValueError(f"expected chain ID {CHAIN_ID}, got {chain_id}")
        end_block = config.end_block
        if end_block is None:
            end_block = int(await rpc(session, config.rpc_url, "eth_blockNumber", []), 16)
        end_block -= (end_block - config.start_block) % config.block_step

        records: list[dict] = []
        base_metadata = None
        if config.base_history:
            base_metadata, records = read_history(config.base_history)
            validate_base(base_metadata, records, config)

        fetch_start = config.start_block
        if records:
            fetch_start = records[-1]["block"] + config.block_step
        blocks = list(range(fetch_start, end_block + 1, config.block_step))
        chunks = [
            blocks[index : index + config.samples_per_request]
            for index in range(0, len(blocks), config.samples_per_request)
        ]
        semaphore = asyncio.Semaphore(config.concurrency)
        started = time.monotonic()
        fetched: list[dict] = []
        pending = {asyncio.create_task(fetch_chunk(session, semaphore, config, chunk)): chunk for chunk in chunks}
        try:
            for completed, task in enumerate(asyncio.as_completed(pending), 1):
                fetched.extend(await task)
                if completed % 25 == 0 or completed == len(chunks):
                    elapsed = time.monotonic() - started
                    print(
                        f"chunks={completed}/{len(chunks)} new_records={len(fetched)} " f"elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

        records.extend(fetched)
        records.sort(key=lambda item: item["block"])
        if not records:
            raise ValueError("no valid observations returned")
        if len({item["block"] for item in records}) != len(records):
            raise ValueError("duplicate blocks after base-history extension")

        pin = await rpc(session, config.rpc_url, "eth_getBlockByNumber", [hex(end_block), False])
        code_hashes = {}
        for label, address in (("lp_pool", LP_POOL), ("bridge_pool", BRIDGE_POOL), ("crvusd_aggregator", CRVUSD_AGG)):
            code = await rpc(session, config.rpc_url, "eth_getCode", [address, hex(end_block)])
            code_hashes[label] = hashlib.sha256(bytes.fromhex(str(code)[2:])).hexdigest()

    metadata = {
        "schema": 2,
        "chain_id": CHAIN_ID,
        "start_block_requested": config.start_block,
        "end_block_requested": end_block,
        "block_step": config.block_step,
        "record_count": len(records),
        "first_block": records[0]["block"],
        "last_block": records[-1]["block"],
        "first_timestamp": records[0]["timestamp"],
        "last_timestamp": records[-1]["timestamp"],
        "pin": {
            "block_number": end_block,
            "block_hash": pin["hash"],
            "block_timestamp": int(pin["timestamp"], 16),
            "observed_at_utc": datetime.now(UTC).isoformat(),
        },
        "contracts": {
            "lp_pool": LP_POOL,
            "bridge_pool": BRIDGE_POOL,
            "crvusd_aggregator": CRVUSD_AGG,
        },
        "runtime_code_sha256": code_hashes,
        "extension": {
            "base_history": config.base_history.name if config.base_history else None,
            "base_record_count": base_metadata["record_count"] if base_metadata else 0,
            "fetched_record_count": len(fetched),
        },
    }
    return records, metadata


def write_output(path: Path, records: list[dict], metadata: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"metadata": metadata}, sort_keys=True, separators=(",", ":")) + "\n")
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    path.write_bytes(raw.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-history", type=Path)
    parser.add_argument("--start-block", type=int, default=START_BLOCK)
    parser.add_argument("--end-block", type=int)
    parser.add_argument("--block-step", type=int, default=BLOCK_STEP)
    parser.add_argument("--samples-per-request", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args()
    if args.block_step <= 0 or args.samples_per_request <= 0 or args.concurrency <= 0 or args.request_delay < 0:
        raise SystemExit("block-step, samples-per-request and concurrency must be positive; delay cannot be negative")
    config = Config(
        rpc_url=args.rpc_url,
        output=args.output,
        base_history=args.base_history,
        start_block=args.start_block,
        end_block=args.end_block,
        block_step=args.block_step,
        samples_per_request=args.samples_per_request,
        concurrency=args.concurrency,
        request_delay=args.request_delay,
    )
    records, metadata = asyncio.run(collect(config))
    digest = write_output(config.output, records, metadata)
    print(json.dumps({"metadata": metadata, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
