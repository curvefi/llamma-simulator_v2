"""Compare ordered losses and timings against a Git revision; no market data needed.

Run: python benchmarks/benchmark_amm.py --baseline f18e123 --windows 400
Optional: --modules before_amm after_amm benchmarks compiled copies of the same code.
"""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
import platform
import random
import statistics
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_source(source, name):
    module = SimpleNamespace()
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def source_at(ref, path):
    if ref is None:
        return (ROOT / path).read_text()
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)


def loss_function(amm, source):
    tree = ast.parse(source)
    simulator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Simulator")
    method = next(
        node for node in simulator.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_loss"
    )
    # Execute the existing method unchanged, without importing unrelated data providers.
    module = ast.Module(body=[method], type_ignores=[])
    scope = {"LendingAMM": amm, "datetime": datetime, "logger": logging.getLogger("benchmark")}
    exec(compile(module, "<Simulator.calculate_loss>", "exec"), scope)
    return scope["calculate_loss"]


def workload(count):
    rng = random.Random(20260930)
    history = []
    for i in range(4000):
        opening = 1 + 0.02 * math.sin(i / 43) + 0.003 * math.sin(i / 3)
        closing = opening * (1 + 0.002 * math.sin(i))
        oracle = 1 + 0.02 * math.sin((i - 2) / 43)
        history.append(
            [
                1700000040 + 60 * i,
                opening,
                max(opening, closing) * 1.002,
                min(opening, closing) * 0.998,
                closing,
                0.0,
                oracle,
            ]
        )
    tasks = []
    for i in range(count):
        a = [10, 100, 300, 600][i % 4]
        fee = [0.001, 0.005][(i // 4) % 2]
        start = rng.randrange(3900)
        length = [30, 60][(i // 8) % 2]
        tasks.append((a, fee, history[start : start + length]))
    return tasks


def run(loss, context, tasks):
    return [
        loss(context, a, fee, [row[:6] for row in rows], [row[6] for row in rows], 4, 0.25) for a, fee, rows in tasks
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--windows", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--modules", nargs=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = "simulator/amm/lending_amm.py"
    sources = [source_at(args.baseline, path), source_at(None, path)]
    amms = (
        [importlib.import_module(name) for name in args.modules]
        if args.modules
        else [load_source(source, label) for source, label in zip(sources, ["baseline", "candidate"])]
    )
    if args.modules:
        sources = [Path(module.__file__).with_name(module.__name__ + ".py").read_text() for module in amms]
    simulator_sources = [source_at(ref, "simulator/amm/simulator.py") for ref in (args.baseline, None)]
    liquidity = load_source(source_at(args.baseline, "simulator/amm/intitial_liquidity.py"), "liquidity")
    context = SimpleNamespace(
        initial_liquidity_class=liquidity.ConstantInitialLiquidity,
        external_fee=0.0005,
        log_enabled=False,
        verbose=False,
    )
    functions = [
        getattr(module, "calculate_loss", None) or loss_function(module.LendingAMM, source)
        for module, source in zip(amms, simulator_sources)
    ]
    tasks = workload(args.windows)
    expected = run(functions[0], context, tasks)
    assert all(math.isfinite(value) for value in expected)
    for _ in range(args.warmup):
        for index, function in enumerate(functions):
            assert run(function, context, tasks) == expected, f"Ordered losses changed during warmup: variant {index}"
    timings = [[], []]
    for trial in range(args.repeats):
        for index in [0, 1] if trial % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            actual = run(functions[index], context, tasks)
            timings[index].append(time.perf_counter() - start)
            assert actual == expected, "Ordered losses changed"
    medians = list(map(statistics.median, timings))
    receipt = {
        "python": platform.python_implementation() + " " + platform.python_version(),
        "machine": platform.machine(),
        "baseline": args.baseline,
        "windows": len(tasks),
        "source_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in sources],
        "modules": args.modules,
        "simulator_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in simulator_sources],
        "warmup_batches": args.warmup,
        "seconds": timings,
        "median_seconds": medians,
        "speedup": medians[0] / medians[1],
        "ordered_losses_equal": True,
        "loss_sha256": hashlib.sha256(struct.pack(f"<{len(expected)}d", *expected)).hexdigest(),
    }
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
