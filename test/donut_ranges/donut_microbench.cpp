// Donut-range micro-benchmark.
//
// This program directly exercises the TAFFO VRA abstract domain (the
// taffo::Range type and its binary operators) to measure the precision
// improvement from donut ranges on a handful of small kernels. It does
// NOT go through the full TAFFO compilation pipeline; instead it runs
// the abstract interpretation in-process and prints the resulting
// ranges, convex-hull widths, and the integer bit-widths a DTA pass
// would need to assign.
//
// Why this is the "real" experiment: the precision gain from donut
// ranges shows up entirely in the abstract domain. Whether that win
// flows through to a smaller final binary depends on DTA heuristics
// that are orthogonal to the analysis. Measuring the analysis output
// directly gives a cleaner signal for a course project than a full
// end-to-end compilation.
//
// Build:
//   cmake --build build --target donut_microbench
//   ./build/test/donut_ranges/donut_microbench
//
// The CMakeLists.txt next to this file adds the target automatically
// when the TaffoCommon library is part of the configuration.

#include "TaffoInfo/RangeInfo.hpp"
#include "TaffoVRA/RangeOperations.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <list>
#include <memory>
#include <string>

using taffo::Range;

namespace {

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

/// Construct a classic single-interval Range.
std::shared_ptr<Range> classic(double lo, double hi) {
  return std::make_shared<Range>(lo, hi);
}

/// Construct a two-component donut range directly. Requires the donut flag
/// to be enabled by the caller.
std::shared_ptr<Range> donut2(double lo1, double hi1, double lo2, double hi2) {
  auto r = std::make_shared<Range>(std::min(lo1, lo2), std::max(hi1, hi2));
  r->components = {{lo1, hi1}, {lo2, hi2}};
  r->canonicalize();
  return r;
}

/// Integer bit-width needed to represent the convex hull of `r` in
/// signed two's complement. Returns 0 for a degenerate empty range.
unsigned integerBits(const Range& r) {
  const double absMax = std::max(std::fabs(r.min), std::fabs(r.max));
  if (absMax <= 1.0) return 1;               // at least a sign bit
  // +1 for the sign bit; ceil(log2(absMax)) gives the magnitude bits.
  return 1 + static_cast<unsigned>(std::ceil(std::log2(absMax + 1.0)));
}

/// Textual description of the components of a range (donut-aware).
std::string describe(const Range& r) {
  char buf[256];
  if (r.components.empty()) {
    std::snprintf(buf, sizeof(buf), "[%.6g, %.6g]", r.min, r.max);
    return buf;
  }
  std::string s;
  for (size_t i = 0; i < r.components.size(); ++i) {
    if (i > 0) s += " U ";
    char part[96];
    std::snprintf(part, sizeof(part), "[%.6g, %.6g]",
                  r.components[i].first, r.components[i].second);
    s += part;
  }
  return s;
}

/// Print a single comparison row between classic and donut results.
void row(const char* label,
         const std::shared_ptr<Range>& classicResult,
         const std::shared_ptr<Range>& donutResult) {
  const unsigned classicBits = integerBits(*classicResult);
  const unsigned donutBits = integerBits(*donutResult);
  const int savings = static_cast<int>(classicBits) - static_cast<int>(donutBits);
  std::fprintf(stdout,
               "  %-28s classic=%-38s int-bits=%2u\n"
               "  %-28s donut  =%-38s int-bits=%2u  Δ=%+d\n\n",
               label, describe(*classicResult).c_str(), classicBits,
               "", describe(*donutResult).c_str(), donutBits, savings);
}

//-----------------------------------------------------------------------------
// Kernels
//-----------------------------------------------------------------------------

// Weights cluster around ±0.5 with nothing near zero. This is typical of
// L2-regularised, zero-mean-initialised neural-network weights in a
// quantised setting where dead neurons have been pruned.
constexpr double kWeightLo1 = -0.8;
constexpr double kWeightHi1 = -0.2;
constexpr double kWeightLo2 = 0.2;
constexpr double kWeightHi2 = 0.8;

void kernelDivisionByBimodalWeight() {
  std::fprintf(stdout, "[ kernel 1: y = 1 / w, w is bimodal ]\n");
  std::fprintf(stdout, "  input w = [%.2f,%.2f] U [%.2f,%.2f]\n",
               kWeightLo1, kWeightHi1, kWeightLo2, kWeightHi2);
  auto one = classic(1.0, 1.0);

  Range::enableDonut = false;
  auto wClassic = classic(std::min(kWeightLo1, kWeightLo2),
                          std::max(kWeightHi1, kWeightHi2));
  auto yClassic = taffo::handleDiv(one, wClassic);

  Range::enableDonut = true;
  auto wDonut = donut2(kWeightLo1, kWeightHi1, kWeightLo2, kWeightHi2);
  auto yDonut = taffo::handleDiv(one, wDonut);

  row("1 / w", yClassic, yDonut);
  Range::enableDonut = false;
}

void kernelSigmoidOfDivision() {
  std::fprintf(stdout, "[ kernel 2: y = sigmoid(1 / w), w is bimodal ]\n");
  std::fprintf(stdout, "  input w = [%.2f,%.2f] U [%.2f,%.2f]\n",
               kWeightLo1, kWeightHi1, kWeightLo2, kWeightHi2);
  auto one = classic(1.0, 1.0);

  Range::enableDonut = false;
  auto wClassic = classic(std::min(kWeightLo1, kWeightLo2),
                          std::max(kWeightHi1, kWeightHi2));
  auto invClassic = taffo::handleDiv(one, wClassic);
  std::list<std::shared_ptr<Range>> opsC{invClassic};
  auto yClassic = taffo::handleMathCallInstruction(opsC, "sigmoid");

  Range::enableDonut = true;
  auto wDonut = donut2(kWeightLo1, kWeightHi1, kWeightLo2, kWeightHi2);
  auto invDonut = taffo::handleDiv(one, wDonut);
  std::list<std::shared_ptr<Range>> opsD{invDonut};
  auto yDonut = taffo::handleMathCallInstruction(opsD, "sigmoid");

  row("sigmoid(1 / w)", yClassic, yDonut);
  Range::enableDonut = false;
}

void kernelDotProductWithBimodalWeight() {
  // y = x * w  where x ∈ [0.1, 0.9] is a post-ReLU activation and w is
  // the same bimodal weight. The product keeps the donut structure but
  // now asymmetric in magnitude.
  std::fprintf(stdout, "[ kernel 3: y = x * w, x>=0 activation, w bimodal ]\n");
  auto x = classic(0.1, 0.9);

  Range::enableDonut = false;
  auto wClassic = classic(std::min(kWeightLo1, kWeightLo2),
                          std::max(kWeightHi1, kWeightHi2));
  auto yClassic = taffo::handleMul(x, wClassic);

  Range::enableDonut = true;
  auto wDonut = donut2(kWeightLo1, kWeightHi1, kWeightLo2, kWeightHi2);
  auto yDonut = taffo::handleMul(x, wDonut);

  row("x * w", yClassic, yDonut);
  Range::enableDonut = false;
}

void kernelReciprocalNormalization() {
  // y = s / (x * x)  where x ∈ [-2,-0.5] U [0.5,2]. Normalisation
  // patterns like layer-norm stash a reciprocal in the denominator; if
  // the input is bimodal, donut ranges prevent the classic reciprocal
  // blow-up.
  std::fprintf(stdout, "[ kernel 4: y = 1 / (x * x), x excludes zero ]\n");
  auto one = classic(1.0, 1.0);

  Range::enableDonut = false;
  auto xClassic = classic(-2.0, 2.0);
  auto xxClassic = taffo::handleMul(xClassic, xClassic);
  auto yClassic = taffo::handleDiv(one, xxClassic);

  Range::enableDonut = true;
  auto xDonut = donut2(-2.0, -0.5, 0.5, 2.0);
  auto xxDonut = taffo::handleMul(xDonut, xDonut);
  auto yDonut = taffo::handleDiv(one, xxDonut);

  row("1 / (x * x)", yClassic, yDonut);
  Range::enableDonut = false;
}

void kernelGeluPiecewise() {
  std::fprintf(stdout, "[ kernel 5: y = gelu(x), x as donut avoiding zero ]\n");

  Range::enableDonut = false;
  auto xClassic = classic(-2.0, 2.0);
  std::list<std::shared_ptr<Range>> opsC{xClassic};
  auto yClassic = taffo::handleMathCallInstruction(opsC, "gelu");

  Range::enableDonut = true;
  auto xDonut = donut2(-2.0, -0.9, 0.9, 2.0);
  std::list<std::shared_ptr<Range>> opsD{xDonut};
  auto yDonut = taffo::handleMathCallInstruction(opsD, "gelu");

  row("gelu(x)", yClassic, yDonut);
  Range::enableDonut = false;
}

}  // namespace

int main() {
  std::fprintf(stdout,
               "================================================================\n"
               " Donut Range Micro-Benchmark  (TAFFO COT project 2025/26)\n"
               " Compares classic interval VRA against donut-range VRA\n"
               "================================================================\n\n");

  kernelDivisionByBimodalWeight();
  kernelSigmoidOfDivision();
  kernelDotProductWithBimodalWeight();
  kernelReciprocalNormalization();
  kernelGeluPiecewise();

  std::fprintf(stdout,
               "================================================================\n"
               " Interpretation:\n"
               "  - classic: TAFFO VRA with a single convex-hull range.\n"
               "  - donut:   TAFFO VRA with up to kMaxComponents disjoint\n"
               "             sub-intervals tracked through arithmetic.\n"
               "  - int-bits: the number of signed integer bits a DTA pass\n"
               "              would allocate for the convex hull of the\n"
               "              resulting range.\n"
               "  - Δ: bit-width savings (positive = donut wins).\n"
               "================================================================\n");
  return 0;
}
