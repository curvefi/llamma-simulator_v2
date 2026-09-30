# AMM replay benchmark

From the repository root:

```sh
python benchmarks/benchmark_amm.py \
  --baseline f18e1231bdc47798314463d1595c69b9dcaacd6f \
  --windows 2000 --warmup 2 --repeats 7
```

This uses only the standard library. It executes each revision's `LendingAMM`
and existing `Simulator.calculate_loss` method, extracting the method to avoid
importing unrelated data providers. Financial calculations are not copied into
the benchmark.

The fixed-seed workload combines A=10/100/300/600, fees=0.001/0.005, 30/60-minute
windows, four bands, a 0.25 dynamic multiplier and a 0.0005 external fee.
Logging and verbose output are disabled. Trials alternate execution order and
require every ordered loss to match exactly, including during warmup. Output
includes all timings and source/output hashes; `--output result.json` saves it.
Collection, preparation, multiprocessing and aggregation are outside this
measurement. Speedups depend on the workload and machine.

On macOS arm64, CPython 3.11.15, the median for 2,000 windows fell from
**3.214 s to 1.899 s (1.69×)**. The runtime patch only moves the existing
empty-band return before price calculations and date formatting inside the
logging branch. Band coverage, funded-band arithmetic and summation order are
unchanged.

In a separate 800-window attribution run, median times were 1.260 s for the
baseline, 0.825 s with the empty-band return moved, and 0.742 s with both changes.
Band/power caches did not provide a consistent additional benefit. An
oracle-view cache helped CPython but regressed the PyPy workload; neither cache
is included.

PyPy 7.3.21 on this host failed repeatability with the unchanged baseline alone:
one of 800 losses changed by 0.0004841334007219533 between repeated replays.
The benchmark intentionally fails rather than relaxing equality. Before/after
losses matched exactly with `pypy3.11 --jit off`; no PyPy JIT speedup or parity
claim is made here.
