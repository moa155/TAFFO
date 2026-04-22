<img src="doc/logo/TAFFO-logo-black.png" alt="TAFFO" width=30%>

> **CTO 2025/26 project fork — TAFFO with Donut Ranges + NN Activation Range Models**
>
> This fork extends upstream TAFFO with two new analyses, delivered as a
> single combined contribution for the *Code Transformation and
> Optimization* course at Politecnico di Milano (A.Y. 2025/26, Prof. G.
> Agosta). Author: Mohamed Z. M. Mandour (person code 10736813).
>
> Full written report: **[`doc/report.pdf`](doc/report.pdf)** (≈ 237 KiB,
> 9 sections + 2 appendices, 85 passing test assertions, 5-kernel
> precision micro-benchmark, two end-to-end benchmarks driven through
> the TAFFO driver).
>
> Design note with the lattice / arithmetic / widening definitions:
> **[`doc/donut_ranges_design.md`](doc/donut_ranges_design.md)**.
>
> All project code lives under [`test/donut_ranges/`](test/donut_ranges/)
> (tests, scripts, benchmarks) and is touched in focused commits in
> [`lib/TaffoCommon/`](lib/TaffoCommon/),
> [`lib/TaffoInitializer/`](lib/TaffoInitializer/),
> [`lib/TaffoVRA/`](lib/TaffoVRA/), and
> [`lib/TaffoConversion/`](lib/TaffoConversion/). Run
> `git log upstream/master..master` to see the 20 project-specific
> commits in dependency order.

### New contributions in this fork

1. **Donut ranges in VRA** — `taffo::Range` now represents each SSA
   value as a sorted canonicalised union of up to `kMaxComponents`
   disjoint closed intervals. All binary arithmetic operators, the
   self-square fast path, and `getUnionRange` propagate the
   component list; division gains a zero-exclusion fast path that
   skips the `DIV_EPS` nudge when every divisor component strictly
   avoids zero. Details in `lib/TaffoCommon/TaffoInfo/RangeInfo.{hpp,cpp}`
   and `lib/TaffoVRA/TaffoVRA/RangeOperations.cpp`.
2. **Range models for 8 NN activation functions** — sigmoid, ReLU,
   leaky ReLU, ELU, softplus, GELU, SiLU/Swish, plus a cleaned-up
   tanh handler. Non-monotonic kernels (GELU, SiLU) use a closed-form
   case-split around mpmath-certified minimum constants (60-digit
   precision, 1000-point grid soundness check). See
   `lib/TaffoVRA/TaffoVRA/RangeOperationsCallWhitelist.cpp` and
   `test/donut_ranges/verify_activation_bounds.py`.
3. **Source-level donut annotation syntax** — the new
   `scalar(range_union((a1,b1),(a2,b2),...))` lets users declare
   donut ranges directly at source level, parsed by
   `lib/TaffoInitializer/AnnotationParser.cpp`. The annotation is
   backward-compatible: with the flag off, VRA uses the convex hull
   and ignores the components, but the metadata round-trips.
4. **Opt-in flag** — `-Xvra -vra-donut-ranges` turns the feature on;
   the default is off, so existing TAFFO users see zero behavioural
   change.
5. **Pre-existing TAFFO Conversion-pass fixes** uncovered while
   running the project benchmarks through the full driver:
   a missing insertion point in `convertStore` and a
   float-metadata / LLVM-type mismatch in `genConvertConvToConv`.
   Both are unrelated to donut ranges but block any NN-shaped user
   code; fixed in `lib/TaffoConversion/Conversion/`.

### Quick start (macOS, Apple Silicon)

```shell
cd "$HOME/TAFFO"
cmake -B cmake-build-debug -S . \
      -DLLVM_DIR=/opt/homebrew/opt/llvm@18/lib/cmake/llvm
cmake --build cmake-build-debug --target donut_range_selftest \
      donut_arith_test donut_microbench Taffo -j
cmake --install cmake-build-debug --prefix "$PWD/install"

# 85-assertion unit-test suite
./cmake-build-debug/test/donut_ranges/donut_range_selftest
./cmake-build-debug/test/donut_ranges/donut_arith_test

# 5-kernel precision micro-benchmark
./cmake-build-debug/test/donut_ranges/donut_microbench

# End-to-end range_union annotation smoke test (8 assertions)
test/donut_ranges/run_donut_annotated.sh

# Toy MLP through the full pipeline (classic vs donut)
test/donut_ranges/run_donut_mlp.sh

# mpmath certificate for GELU / SiLU minima
python3 test/donut_ranges/verify_activation_bounds.py
```

The reference machine for the numbers in the report is an
Apple Silicon MacBook Pro running Homebrew LLVM 18.1.8, macOS 26.4.
Linux with LLVM 18 also works (use the upstream instructions below
and replace `LLVM_DIR` accordingly).

---

## Upstream TAFFO documentation

TAFFO *(Tuning Assistant for Floating Point to Fixed Point Optimization)* is a precision-tuning framework to replace floating point operations with fixed point operations.

It is based on LLVM and has been tested on Linux and, as part of the
CTO 2025/26 project, on macOS with Homebrew LLVM 18 (Apple Silicon).

## How to use TAFFO

Taffo currently ships as 5 LLVM plugins, each one of them containing one LLVM optimization or analysis pass:
 - TaffoTypeDeducer (Small wrapper of the TDA pass https://github.com/NiccoloN/TypeDeductionAnalysis)
 - TaffoInitializer (Initialization pass)
 - TaffoVRA (Value Range Analysis pass)
 - TaffoDTA (Data Type Allocation pass)
 - TaffoConversion (Conversion pass)

To execute TAFFO, a simple frontend is provided named `taffo`, which can be substituted to `clang` in order to compile or link executables.
Behind the scenes, it uses the LLVM `opt` tool to load one pass at a time and run it on LLVM IR source files.

### 1: Build and install TAFFO

Create a build directory, compile and install TAFFO.
You can either install TAFFO to the standard location of `/usr/local`, or you can install it to any other location of your choice.
In the latter case you will have to add that location to your PATH.

If you have multiple LLVM versions installed, and you want to link TAFFO to a specific one, set the `LLVM_DIR` environment variable to the install-prefix of the correct LLVM version beforehand.

At the moment TAFFO supports LLVM 18 in the master branch and LLVM 14/15 in the llvm-15 branch. No other version is currently supported.
Moreover, LLVM plugins compiled for a given major version of LLVM cannot be loaded by any other version. Therefore, it is not a good idea to redistribute TAFFO as a binary.
If you are building LLVM from sources, you must configure it with `-DLLVM_BUILD_LLVM_DYLIB=ON` and `-DLLVM_LINK_LLVM_DYLIB=ON` for the TAFFO build to succeed.

The following are the minimal commands required for compiling TAFFO on a typical Linux distribution.

```shell
cd /path/to/the/location/of/TAFFO
export LLVM_DIR=/usr/lib/llvm-18 # optional
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```

If you want to modify TAFFO or see the debug logs you need to also build LLVM in debug mode first.
You are encouraged to follow our guide: [Building LLVM](doc/BuildingLLVM.md)

### 2: Modify and test the application

Modify the application to insert annotations on the appropriate variable declarations, then use `taffo` to compile your application.

```shell
<editor> program.c
[...]
taffo -O3 -o program-taffo program.c
```

See the annotation syntax documentation or the examples in `test/0_simple` to get an idea on how to write annotations.
You can also test TAFFO without adding annotations, which will produce the same results as using `clang` as a compiler/linker instead of `taffo`.

Note that there is no `taffo++`; C++ source files are autodetected by the file extension instead.

## How to build and run tests and benchmarks

Optionally, create and activate a python virtual environment in the repository root directory:
```shell
python3 -m venv ./venv
source .venv/bin/activate
```

Install python requirements
```shell
pip install -r requirements.txt
```

Then, access the build directory of TAFFO and run all the tests and benchmarks:
```shell
cd test
ctest
```

To run test suites individually and inspect the results, access the test-suite directory of interest and use the python runner, for example:
```shell
cd test/polybench
./run.py
```
Add the option ```-debug``` to the runner to be able to inspect TAFFO's temporary files and debug log, generated in each benchmark's directory.

You can also run a single test at a time, for example:
```shell
cd test/polybench
./run.py -only correlation
```

## How to debug TAFFO

When invoked with the ```-debug``` option, TAFFO launches ```opt``` once for each of its passes with commands like:
```
/path/to/opt -load /path/to/Taffo.so --load-pass-plugin=/path/to/Taffo.so --passes=no-op-module,typededucer                     --stats --debug-only=tda,taffo-common,taffo-typededucer   -S -o out.ll in.ll
/path/to/opt -load /path/to/Taffo.so --load-pass-plugin=/path/to/Taffo.so --passes=no-op-module,taffoinit                       --stats --debug-only=taffo-common,taffo-init              -S -o out.ll in.ll
/path/to/opt -load /path/to/Taffo.so --load-pass-plugin=/path/to/Taffo.so --passes=no-op-module,function(taffomem2reg),taffovra --stats --debug-only=taffo-common,taffo-mem2reg,taffo-vra -S -o out.ll in.ll
/path/to/opt -load /path/to/Taffo.so --load-pass-plugin=/path/to/Taffo.so --passes=no-op-module,taffodta,globaldce              --stats --debug-only=taffo-common,taffo-dta               -S -o out.ll in.ll
/path/to/opt -load /path/to/Taffo.so --load-pass-plugin=/path/to/Taffo.so --passes=no-op-module,taffoconv,globaldce,dce         --stats --debug-only=taffo-common,taffo-conv              -S -o out.ll in.ll
```

You can find the real commands in the debug log of each test, in the lines just before the start of each pass, marked by ```[NameOfThePass]``` (e.g. ```[ConversionPass]```).
For example, in my case I get these for the benchmark ```correlation``` inside polybench:
```
/home/nico/llvm/llvm-18-install/bin/opt -load /home/nico/taffo/taffo-install/lib/Taffo.so --load-pass-plugin=/home/nico/taffo/taffo-install/lib/Taffo.so --passes=no-op-module,typededucer                     --stats --debug-only=tda,taffo-common,taffo-typededucer   -temp-dir=./taffo_temp -S -o ./taffo_temp/correlation-taffo.1.taffotmp.ll ./taffo_temp/correlation-taffo.0.taffotmp.ll
/home/nico/llvm/llvm-18-install/bin/opt -load /home/nico/taffo/taffo-install/lib/Taffo.so --load-pass-plugin=/home/nico/taffo/taffo-install/lib/Taffo.so --passes=no-op-module,taffoinit                       --stats --debug-only=taffo-common,taffo-init              -temp-dir=./taffo_temp -S -o ./taffo_temp/correlation-taffo.2.taffotmp.ll ./taffo_temp/correlation-taffo.1.taffotmp.ll
/home/nico/llvm/llvm-18-install/bin/opt -load /home/nico/taffo/taffo-install/lib/Taffo.so --load-pass-plugin=/home/nico/taffo/taffo-install/lib/Taffo.so --passes=no-op-module,function(taffomem2reg),taffovra --stats --debug-only=taffo-common,taffo-mem2reg,taffo-vra -temp-dir=./taffo_temp -S -o ./taffo_temp/correlation-taffo.3.taffotmp.ll ./taffo_temp/correlation-taffo.2.taffotmp.ll
/home/nico/llvm/llvm-18-install/bin/opt -load /home/nico/taffo/taffo-install/lib/Taffo.so --load-pass-plugin=/home/nico/taffo/taffo-install/lib/Taffo.so --passes=no-op-module,taffodta,globaldce              --stats --debug-only=taffo-common,taffo-dta               -temp-dir=./taffo_temp -S -o ./taffo_temp/correlation-taffo.4.taffotmp.ll ./taffo_temp/correlation-taffo.3.taffotmp.ll
/home/nico/llvm/llvm-18-install/bin/opt -load /home/nico/taffo/taffo-install/lib/Taffo.so --load-pass-plugin=/home/nico/taffo/taffo-install/lib/Taffo.so --passes=no-op-module,taffoconv,globaldce,dce         --stats --debug-only=taffo-common,taffo-conv              -temp-dir=./taffo_temp -S -o ./taffo_temp/correlation-taffo.5.taffotmp.ll ./taffo_temp/correlation-taffo.4.taffotmp.ll
```

### VSCode:
To debug a TAFFO pass in VSCode, use the following launch configuration template, adapting it to run the correct command for your case:
```json
{
    "type": "lldb",
    "request": "launch",
    "name": "launch-config-name",
    "program": "/path/to/opt",
    "args": [
        "-load",
        "/path/to/Taffo.so",
        "...",
        "output_file",
        "input_file"
    ],
    "cwd": "/path/to/specific/test/dir",
    "initCommands": [
        "settings set target.process.follow-fork-mode child"
    ],
    "sourceLanguages": [
        "c"
    ]
}
```

### CLion:
To debug a TAFFO pass in CLion, insert the line `settings set target.load-cwd-lldbinit true` in file `.lldbinit` in you home directory.
Then, create a run configuration for a CMake application with target Taffo, executable ```opt```,
and adjust the program arguments and working directory to run the correct command for your case.
Finally, you can also specify to build and install TAFFO, before execution.

<img src="doc/images/CLionDebugConfiguration.png" alt="TAFFO" width=50%>
