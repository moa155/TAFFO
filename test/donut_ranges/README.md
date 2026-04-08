# Donut Ranges — Tests & Micro-Benchmark

This directory hosts the correctness test and the micro-benchmark for the
COT 2025/26 project "Donut Ranges in VRA + Range Models for NN Activation
Functions". See `doc/donut_ranges_design.md` for the design note and
`doc/report.md` for the full project report.

## Files

| File | Purpose |
|---|---|
| `donut_range_selftest.cpp` | Self-contained correctness tests for `taffo::Range` donut extensions (canonicalisation, serialisation round-trip, flag-off compatibility). Plain C++ + `assert`, no gtest. |
| `donut_microbench.cpp` | Measures the precision improvement from donut-range VRA on 5 small kernels (reciprocal of bimodal weight, sigmoid of reciprocal, weight–activation product, layer-norm-style normalisation, and GELU piecewise). |
| `CMakeLists.txt` | Hooks both binaries into the TAFFO build; automatically skipped if `TaffoCommon` / `obj.TaffoVRA` targets are not in the configuration. |
| `README.md` | This file. |

## Building

The targets are added by `test/CMakeLists.txt` whenever `TaffoCommon` is
part of the configuration, which happens for any normal TAFFO build.

```bash
cd /Users/mohamed/CLionProjects/PoliMI/TAFFO
cmake -B build -S . -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
cmake --build build --target donut_range_selftest
cmake --build build --target donut_microbench
```

The exact CMake invocation is the same as for any other TAFFO target; only
the two executables listed above are specific to this project.

## Running

```bash
./build/test/donut_ranges/donut_range_selftest
./build/test/donut_ranges/donut_microbench
```

`donut_range_selftest` prints one line per test section and a final
summary `X passed, Y failed`. A non-zero exit status indicates a test
failure.

`donut_microbench` prints one row per kernel showing, side by side:

* the resulting range under **classic** interval VRA,
* the resulting range under **donut-range** VRA,
* the signed-integer bit-width a DTA pass would assign to each,
* the saving Δ (positive = donut wins).

## Expected results (computed by hand)

If the implementation is correct, the micro-benchmark should print
numbers equivalent to the following table. Integer bit-widths include
the sign bit.

| Kernel | Classic range | Donut range | Classic int-bits | Donut int-bits | Δ |
|---|---|---|---:|---:|---:|
| `1 / w` with `w ∈ [-0.8,-0.2] ∪ [0.2,0.8]` | `[-1e8, 1e8]` (DIV_EPS blow-up) | `[-5, -1.25] ∪ [1.25, 5]` | 28 | 4 | **+24** |
| `sigmoid(1 / w)` | `[~0, ~1]` | `[~0.007, ~0.22] ∪ [~0.78, ~0.993]` | 1 | 1 | 0 (but hull coverage ≈ 44% vs 100%) |
| `x * w` with `x ∈ [0.1, 0.9]`, `w` bimodal | `[-0.72, 0.72]` | `[-0.72, -0.02] ∪ [0.02, 0.72]` | 1 | 1 | 0 (hole preserved) |
| `1 / (x * x)` with `x ∈ [-2,-0.5] ∪ [0.5, 2]` | `[-1e8, 1e8]` (DIV_EPS blow-up) | `[0.25, 4]` | 28 | 3 | **+25** |
| `gelu(x)` with `x ∈ [-2,-0.9] ∪ [0.9, 2]` | `[-0.1701, ~1.955]` | `[-0.1701, ~-0.164] ∪ [~0.707, ~1.955]` | 2 | 2 | 0 (hole preserved) |

The two headline wins are kernels **1** and **4**: bimodal distributions
feeding into a division step benefit massively from donut tracking because
each sub-interval can be divided exactly, without the
`DIV_EPS = 1e-8` nudge that the classic path uses to avoid literal
division-by-zero.

Kernels 2, 3 and 5 show smaller int-bit deltas but still preserve the
donut hole, which is visible in the printed range. A future DTA extension
that allocates per-component bit-widths can exploit this extra structure.

## Opt-in behaviour

Both binaries exercise the `Range::enableDonut` global flag directly
(equivalent to the `-vra-donut-ranges` LLVM command-line option wired up
in `ValueRangeAnalysisPass.cpp`). With the flag off, every kernel should
degenerate to the classic interval result exactly — no behavioural change
for existing TAFFO users.
