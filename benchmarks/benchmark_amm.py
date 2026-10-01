"""Compare ordered losses and timings against a Git revision; no market data needed.

Run: python benchmarks/benchmark_amm.py --baseline f18e123 --windows 400
"""

import argparse
import ast
import gzip
import hashlib
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


def checked_losses(values, expected=None):
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite loss")
    packed = struct.pack(f"<{len(values)}d", *values)
    if expected is not None and packed != expected:
        raise ValueError("Ordered losses changed")
    return packed


def measure(functions, contexts, tasks, warmup, repeats):
    expected = run(functions[0], contexts[0], tasks)
    if len(expected) != len(tasks):
        raise ValueError("Expected one loss per window")
    packed = checked_losses(expected)
    for _ in range(warmup):
        for function, context in zip(functions, contexts):
            actual = run(function, context, tasks)
            checked_losses(actual, packed)
    timings = [[] for _ in functions]
    for trial in range(repeats):
        order = range(len(functions)) if trial % 2 == 0 else reversed(range(len(functions)))
        for index in order:
            start = time.perf_counter()
            actual = run(functions[index], contexts[index], tasks)
            timings[index].append(time.perf_counter() - start)
            checked_losses(actual, packed)
    medians = list(map(statistics.median, timings))
    return {
        "seconds": timings,
        "median_seconds": medians,
        "speedup_vs_baseline": [medians[0] / value for value in medians],
        "ordered_losses_equal": True,
        "ordered_losses": expected,
        "loss_sha256": hashlib.sha256(packed).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, nargs="+", help="One or more reference revisions")
    parser.add_argument("--candidate", help="Candidate revision; defaults to working-tree source")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--windows", type=int, default=400)
    inputs.add_argument("--workload", type=Path, help="JSON or JSON.gz containing tasks and provenance")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.windows <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("windows and repeats must be positive; warmup must be nonnegative")

    revisions, functions, contexts = [], [], []
    for ref in [*args.baseline, args.candidate]:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{ref or 'HEAD'}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
        sources = {
            name: source_at(revision if ref else None, f"simulator/amm/{name}.py")
            for name in ("lending_amm", "simulator", "intitial_liquidity")
        }
        revisions.append(
            {
                "revision": revision,
                "working_tree": ref is None,
                "source_sha256": {
                    name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()
                },
            }
        )
        amm = load_source(sources["lending_amm"], "lending_amm")
        liquidity = load_source(sources["intitial_liquidity"], "intitial_liquidity")
        functions.append(loss_function(amm.LendingAMM, sources["simulator"]))
        contexts.append(
            SimpleNamespace(
                initial_liquidity_class=liquidity.ConstantInitialLiquidity,
                external_fee=0.0005,
                log_enabled=False,
                verbose=False,
            )
        )

    if args.workload:
        opener = gzip.open if args.workload.suffix == ".gz" else open
        with opener(args.workload, "rt") as file:
            inputs = json.load(file)
        tasks, provenance = inputs["tasks"], inputs["provenance"]
    else:
        tasks, provenance = workload(args.windows), {"kind": "synthetic", "seed": 20260930}
    if not tasks or any(not rows for _, _, rows in tasks):
        parser.error("workload and every replay window must be nonempty")

    receipt = {
        "python": platform.python_implementation() + " " + platform.python_version(),
        "platform": platform.platform(),
        "revisions": revisions,
        "windows": len(tasks),
        "workload_sha256": hashlib.sha256(json.dumps(tasks, separators=(",", ":")).encode()).hexdigest(),
        "provenance": provenance,
        "settings": {
            "bands": 4,
            "dynamic_fee_multiplier": 0.25,
            "external_fee": 0.0005,
            "position_shift": 0,
            "log_enabled": False,
            "verbose": False,
            "initial_liquidity": "ConstantInitialLiquidity",
        },
        "initialization": "Each revision's upstream calculate_loss; no application initialization adapter",
        "warmup_batches": args.warmup,
    }
    receipt.update(measure(functions, contexts, tasks, args.warmup, args.repeats))
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
