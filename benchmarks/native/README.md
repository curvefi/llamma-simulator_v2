# Native replay: readability review

This branch organizes the existing performance experiment for review before further optimization. It is based on Curve master `0bb370f02970c2c1056da8a4faffd0fa9be35575`. It is not a proposal to merge every change as one upstream PR.

## Assessment

The performance gain is substantial, but the combined source is more intrusive than a lightweight compilation layer. Helper extraction and explicit quote snapshots are the strongest candidates for an upstream discussion. Custom storage, global caches, native math wrappers and changed configuration behavior need a higher acceptance bar. Smaller, separately justified patches would be easier to maintain and review.

| Change | Readability and compatibility cost | Assessment |
|---|---|---|
| Extract numeric replay and oracle helpers | Moves existing calculations and adds a public-wrapper/private-helper boundary; indexed float64 inputs are less general than arbitrary Python sequences | Reasonable boundary for native compilation; keep the main replay easy to follow |
| Preserve native math behavior | Adds `_power(...)` calls, a square-root wrapper and a Cython runtime dependency to ordinary Python | Necessary for this measured build's rounding behavior; visibly obscures some equations |
| Cache geometry, powers and prices | Adds module-level mutable state and input-key comparisons | Worth considering only with clear dependencies, invalidation behavior and measured gains |
| Numeric band storage | Replaces `defaultdict` with a custom container and changes indexing to `.read()` / `.write()` throughout accounting | Largest maintenance and aesthetic cost; not a complete dictionary substitute |
| Per-instance oracle settings | Moves configuration defaults from class attributes to instance fields | Changes class-wide override behavior; should not be presented as a neutral refactor |
| Reset and numeric batch replay | Adds state-reset obligations and an eight-column resolved-task interface | Useful for throughput; the batch contract is currently specific to the benchmark |

The underlying accounting remains in Python. The branch adds no handwritten C++ implementation. That does not make every change behaviorally equivalent for every Python caller: native integer limits, custom storage, float64 input conversion, final compiled classes and configuration overrides narrow the supported interface.

The batch entry uses unshifted positions and default opening oracle state. Its docstring defines every input column, including the unused eighth slot retained for benchmark compatibility. Each task supplies its external fee explicitly; neither a successful nor a failed batch changes the simulator's configured fee. Reset clears the previous position bounds as well as balances and oracle memory. This remains a resolved-task interface, not a general replacement for the simulator API. The short declarations under `declarations/` are included so reviewers can inspect the complete native typing assumptions, including final classes and scalar field types.

## Measured result

These measurements describe the combined candidate before the final readability cleanup, not independent speedups for each commit or a fresh benchmark of this branch. Five interleaved trials used the same 11,520 frozen ETH/BTC/LP windows and 518,400 candles on Apple M1 Max. Times are the sum of the three per-market medians; data preparation and startup are excluded.

| Implementation | One worker | Four workers |
|---|---:|---:|
| Upstream Curve Python | 4.9741 s | 1.3521 s |
| Existing upstream Cython build | 1.8729 s | 0.5078 s |
| Fork Python | 5.4536 s | 1.4815 s |
| Fork native Cython | 0.1853 s | 0.0520 s |
| Phil C++ with older accounting | 0.0481 s | 0.0144 s |

The native candidate is 10.11x faster than the existing Cython build with one worker and 9.76x with four. Interpreted Python is approximately 10% slower. That regression matters for an upstream project whose default execution is Python. Phil's older accounting differs in every benchmark window, so its timing is a speed reference rather than an equivalent substitute. No complete application or new calibration sweep is represented here.

Before organizing this history, fork Python matched all 11,520 upstream losses bit for bit. The compiled candidate stayed within the existing `5e-14` absolute loss tolerance (maximum `4.01e-14`). Repeated and one/four-worker outputs were bitwise equal within each backend. Two clean builds produced identical generated C++ and extension binaries on the pinned target.

The source commits are grouped by purpose, with formatting folded into the commit that introduces the code. Final cleanup clarifies the numeric replay helper, removes redundant storage syntax, documents cache boundaries and restores the original configuration comments. Focused batch tests cover unchanged simulator settings on success and failure, reordered tasks, empty inputs, invalid shapes and complete workspace reset. Financial formulas and the replay sequence are unchanged.

Each source step passes its Python tests and the existing 48-window replay/state/error comparison against untouched upstream. Native build and frozen-workload checks qualify the completed branch separately. These are bounded checks, not a proof of every possible Python input.

## Build declarations

`declarations/lending_amm.pxd` and `declarations/simulator.pxd` record the current native type declarations, including the explicit per-task fee argument. An external build package stages them alongside the matching `.py` files, then generates and compiles C++. They are review inputs here, not an automatically enabled build or a second accounting implementation.

The measured environment was CPython 3.11.15, Cython 3.2.4 and Apple Clang 21 with SDK 26.5. Compiler flags were `-std=c++17 -O3 -g0 -ffp-contract=off -fno-fast-math`. The full pinned builder and benchmark evidence remain separate from this source-review branch; the declaration files alone are not a complete reproduction package.

Before an upstream submission, separate broadly useful Python changes from native-specific restrictions, simplify the visible accounting operations where possible, and present the Python performance tradeoff explicitly. Further optimization should wait until this source shape is acceptable.
