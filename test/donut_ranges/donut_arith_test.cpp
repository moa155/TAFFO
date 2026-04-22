// Arithmetic unit tests for donut ranges.
//
// This file covers the binary operators `handleAdd`, `handleSub`,
// `handleMul`, `handleDiv` and `getUnionRange` on both classic and donut
// inputs. It is separate from `donut_range_selftest.cpp` (which only
// tests the Range data structure itself) because the arithmetic is
// defined in TaffoVRA, not TaffoCommon, and the two binaries have
// different link dependencies.
//
// Build:
//   cmake --build cmake-build-debug --target donut_arith_test
//   ./cmake-build-debug/test/donut_ranges/donut_arith_test

#include "TaffoInfo/RangeInfo.hpp"
#include "TaffoVRA/RangeOperations.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <list>
#include <memory>

using taffo::Range;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT_TRUE(cond) do {                                           \
  if (!(cond)) {                                                         \
    std::fprintf(stderr, "  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #cond); \
    ++g_failed;                                                          \
  } else {                                                               \
    ++g_passed;                                                          \
  }                                                                      \
} while (0)

#define EXPECT_NEAR(got, want, tol) do {                                 \
  const double _g = (got);                                               \
  const double _w = (want);                                              \
  if (std::fabs(_g - _w) > (tol)) {                                      \
    std::fprintf(stderr, "  FAIL [%s:%d] %s got=%g want=%g tol=%g\n",    \
                 __FILE__, __LINE__, #got, _g, _w, (double)(tol));       \
    ++g_failed;                                                          \
  } else {                                                               \
    ++g_passed;                                                          \
  }                                                                      \
} while (0)

void section(const char* name) {
  std::fprintf(stdout, "[ %s ]\n", name);
}

auto classic(double lo, double hi) {
  return std::make_shared<Range>(lo, hi);
}

auto donut2(double lo1, double hi1, double lo2, double hi2) {
  auto r = std::make_shared<Range>(std::min(lo1, lo2), std::max(hi1, hi2));
  r->components = {{lo1, hi1}, {lo2, hi2}};
  r->canonicalize();
  return r;
}

// ===== Sanity: classic arithmetic is unchanged when donut is OFF =====

void testClassicAddUnchanged() {
  section("classic path unchanged: handleAdd on non-donut inputs");
  Range::enableDonut = false;
  auto a = classic(1.0, 2.0);
  auto b = classic(3.0, 4.0);
  auto r = taffo::handleAdd(a, b);
  EXPECT_TRUE(r->components.empty());
  EXPECT_NEAR(r->min, 4.0, 1e-12);
  EXPECT_NEAR(r->max, 6.0, 1e-12);
}

void testClassicSubUnchanged() {
  section("classic path unchanged: handleSub on non-donut inputs");
  Range::enableDonut = false;
  auto a = classic(5.0, 10.0);
  auto b = classic(1.0, 3.0);
  auto r = taffo::handleSub(a, b);
  EXPECT_TRUE(r->components.empty());
  EXPECT_NEAR(r->min, 2.0, 1e-12);
  EXPECT_NEAR(r->max, 9.0, 1e-12);
}

void testClassicMulUnchanged() {
  section("classic path unchanged: handleMul on non-donut inputs");
  Range::enableDonut = false;
  auto a = classic(-2.0, 3.0);
  auto b = classic(1.0, 4.0);
  auto r = taffo::handleMul(a, b);
  EXPECT_TRUE(r->components.empty());
  EXPECT_NEAR(r->min, -8.0, 1e-12);
  EXPECT_NEAR(r->max, 12.0, 1e-12);
}

// ===== Donut arithmetic: addition =====

void testDonutAdd() {
  section("donut + donut: addition");
  Range::enableDonut = true;
  auto a = donut2(-0.8, -0.2, 0.2, 0.8);
  auto b = donut2(1.0, 2.0, 5.0, 6.0);
  auto r = taffo::handleAdd(a, b);
  // The Cartesian product yields four intervals:
  //   [-0.8,-0.2]+[1, 2] = [0.2, 1.8]
  //   [-0.8,-0.2]+[5, 6] = [4.2, 5.8]
  //   [ 0.2, 0.8]+[1, 2] = [1.2, 2.8]
  //   [ 0.2, 0.8]+[5, 6] = [5.2, 6.8]
  // [0.2, 1.8] and [1.2, 2.8] overlap and merge to [0.2, 2.8].
  // [4.2, 5.8] and [5.2, 6.8] overlap and merge to [4.2, 6.8].
  EXPECT_TRUE(r->components.size() == 2);
  if (r->components.size() == 2) {
    EXPECT_NEAR(r->components[0].first, 0.2, 1e-12);
    EXPECT_NEAR(r->components[0].second, 2.8, 1e-12);
    EXPECT_NEAR(r->components[1].first, 4.2, 1e-12);
    EXPECT_NEAR(r->components[1].second, 6.8, 1e-12);
  }
  EXPECT_NEAR(r->min, 0.2, 1e-12);
  EXPECT_NEAR(r->max, 6.8, 1e-12);
  Range::enableDonut = false;
}

void testClassicPlusDonut() {
  section("classic + donut: hull must remain a donut");
  Range::enableDonut = true;
  auto a = classic(0.0, 0.1);  // small shift
  auto b = donut2(-1.0, -0.5, 0.5, 1.0);
  auto r = taffo::handleAdd(a, b);
  // [0, 0.1] + [-1, -0.5] = [-1, -0.4]
  // [0, 0.1] + [ 0.5, 1] = [ 0.5, 1.1]
  // Gap remains [-0.4, 0.5] -> donut preserved.
  EXPECT_TRUE(r->components.size() == 2);
  if (r->components.size() == 2) {
    EXPECT_NEAR(r->components[0].first, -1.0, 1e-12);
    EXPECT_NEAR(r->components[0].second, -0.4, 1e-12);
    EXPECT_NEAR(r->components[1].first, 0.5, 1e-12);
    EXPECT_NEAR(r->components[1].second, 1.1, 1e-12);
  }
  Range::enableDonut = false;
}

// ===== Donut arithmetic: multiplication =====

void testDonutMulPositiveActivation() {
  section("donut * classic: hole preserved when multiplier excludes zero");
  Range::enableDonut = true;
  auto x = classic(0.1, 0.9);          // post-ReLU activation
  auto w = donut2(-0.8, -0.2, 0.2, 0.8);
  auto r = taffo::handleMul(x, w);
  // [0.1,0.9] * [-0.8,-0.2] = [-0.72, -0.02]
  // [0.1,0.9] * [ 0.2, 0.8] = [ 0.02,  0.72]
  // Gap (-0.02, 0.02) preserved.
  EXPECT_TRUE(r->components.size() == 2);
  if (r->components.size() == 2) {
    EXPECT_NEAR(r->components[0].first, -0.72, 1e-9);
    EXPECT_NEAR(r->components[0].second, -0.02, 1e-9);
    EXPECT_NEAR(r->components[1].first, 0.02, 1e-9);
    EXPECT_NEAR(r->components[1].second, 0.72, 1e-9);
  }
  Range::enableDonut = false;
}

void testDonutSelfSquareTight() {
  section("donut * donut (self): x*x must be non-negative (regression for §7.5)");
  Range::enableDonut = true;
  auto x = donut2(-2.0, -0.5, 0.5, 2.0);
  auto r = taffo::handleMul(x, x);
  // Mathematical truth: {x^2 : x in [-2,-0.5] U [0.5,2]} = [0.25, 4].
  // The donut-aware self-square fast path must NOT fall through to the
  // Cartesian product (which would produce [-4, -0.25] U [0.25, 4]).
  EXPECT_TRUE(r->components.empty());  // collapses to classic [0.25, 4]
  EXPECT_NEAR(r->min, 0.25, 1e-12);
  EXPECT_NEAR(r->max, 4.0, 1e-12);
  Range::enableDonut = false;
}

// ===== Donut arithmetic: division =====

void testDonutDivByZeroExcludingComponents() {
  section("1 / donut excluding zero: no DIV_EPS blow-up");
  Range::enableDonut = true;
  auto one = classic(1.0, 1.0);
  auto w = donut2(-0.8, -0.2, 0.2, 0.8);
  auto r = taffo::handleDiv(one, w);
  // 1/[-0.8,-0.2] = [-5, -1.25]
  // 1/[ 0.2, 0.8] = [ 1.25,  5]
  EXPECT_TRUE(r->components.size() == 2);
  if (r->components.size() == 2) {
    EXPECT_NEAR(r->components[0].first, -5.0, 1e-9);
    EXPECT_NEAR(r->components[0].second, -1.25, 1e-9);
    EXPECT_NEAR(r->components[1].first, 1.25, 1e-9);
    EXPECT_NEAR(r->components[1].second, 5.0, 1e-9);
  }
  // CRUCIAL: the magnitude must stay BOUNDED by 5, not explode to 1e8.
  EXPECT_TRUE(std::fabs(r->min) <= 10.0);
  EXPECT_TRUE(std::fabs(r->max) <= 10.0);
  Range::enableDonut = false;
}

void testClassicDivStillBlows() {
  section("1 / [-0.8, 0.8] (classic): DIV_EPS path still kicks in");
  Range::enableDonut = false;
  auto one = classic(1.0, 1.0);
  auto w = classic(-0.8, 0.8);
  auto r = taffo::handleDiv(one, w);
  // Classic path has no way to exclude zero -> r is huge.
  EXPECT_TRUE(std::fabs(r->max) >= 1e6);
}

void testDonutDivSubjectCrossesZero() {
  section("donut numerator / (donut excluding zero)");
  Range::enableDonut = true;
  auto x = donut2(-3.0, -1.0, 1.0, 3.0);   // numerator
  auto w = donut2(-0.8, -0.2, 0.2, 0.8);   // denominator, excludes 0
  auto r = taffo::handleDiv(x, w);
  // Every component division is bounded: the absolute magnitude of the
  // result is |x|_max / |w|_min = 3 / 0.2 = 15.
  EXPECT_TRUE(std::fabs(r->min) <= 20.0);
  EXPECT_TRUE(std::fabs(r->max) <= 20.0);
  EXPECT_TRUE(!r->components.empty());
  Range::enableDonut = false;
}

// ===== getUnionRange =====

// ===== Non-monotonic activations: certified GELU / SiLU minima =====

void testGELUCertifiedMinimumStraddling() {
  // Straddling case: the input interval contains x_min ≈ -0.7524614.
  // The certified Y_MIN is -0.1702 (see
  // test/donut_ranges/verify_activation_bounds.py). We require the
  // returned lower bound to be exactly that sound constant, and the
  // upper bound to be the larger of gelu(lo) and gelu(hi).
  section("GELU: straddling interval returns the certified Y_MIN");
  Range::enableDonut = false;
  auto x = classic(-1.0, 0.0);
  std::list<std::shared_ptr<Range>> ops{x};
  auto r = taffo::handleMathCallInstruction(ops, "gelu");
  EXPECT_TRUE(r != nullptr);
  EXPECT_NEAR(r->min, -0.1702, 1e-12);
  EXPECT_NEAR(r->max, 0.0, 1e-3);  // gelu(0) == 0, gelu(-1) ≈ -0.159
}

void testGELUMonotoneDecreasingBranch() {
  // lo = -3, hi = -2. hi < X_MIN_HI = -0.76: entire decreasing branch.
  // Returned range is [gelu(hi), gelu(lo)] ≈ [-0.0454, -0.00404].
  section("GELU: fully-decreasing branch uses monotone case A");
  Range::enableDonut = false;
  auto x = classic(-3.0, -2.0);
  std::list<std::shared_ptr<Range>> ops{x};
  auto r = taffo::handleMathCallInstruction(ops, "gelu");
  EXPECT_TRUE(r != nullptr);
  // No Y_MIN involvement — the reported min should be gelu(-2) ≈ -0.0454.
  EXPECT_NEAR(r->min, -0.04540,  1e-3);
  EXPECT_NEAR(r->max, -0.00405, 1e-3);
}

void testSiLUCertifiedMinimumStraddling() {
  // Input contains x_min ≈ -1.278464. Certified Y_MIN is -0.2786.
  section("SiLU: straddling interval returns the certified Y_MIN");
  Range::enableDonut = false;
  auto x = classic(-2.0, 0.0);
  std::list<std::shared_ptr<Range>> ops{x};
  auto r = taffo::handleMathCallInstruction(ops, "silu");
  EXPECT_TRUE(r != nullptr);
  EXPECT_NEAR(r->min, -0.2786, 1e-12);
  EXPECT_NEAR(r->max, 0.0, 1e-3);  // silu(0) = 0, silu(-2) ≈ -0.238
}

void testSiLUMonotoneIncreasingBranch() {
  // lo = -1, hi = 2. lo > X_MIN = -1.27: fully on the increasing branch.
  section("SiLU: fully-increasing branch uses monotone case B");
  Range::enableDonut = false;
  auto x = classic(-1.0, 2.0);
  std::list<std::shared_ptr<Range>> ops{x};
  auto r = taffo::handleMathCallInstruction(ops, "silu");
  EXPECT_TRUE(r != nullptr);
  // silu(-1) ≈ -0.2689, silu(2) ≈ 1.7616.
  EXPECT_NEAR(r->min, -0.26894, 1e-3);
  EXPECT_NEAR(r->max,  1.76159, 1e-3);
}

void testGELUDonutPreservesHole() {
  // Donut input whose left and right components each hit different
  // monotone branches: [-3, -2] on decreasing, [0.5, 1.5] on increasing.
  // Per-component handler should produce a 2-component donut.
  section("GELU: donut input preserves hole across branches");
  Range::enableDonut = true;
  auto x = donut2(-3.0, -2.0, 0.5, 1.5);
  std::list<std::shared_ptr<Range>> ops{x};
  auto r = taffo::handleMathCallInstruction(ops, "gelu");
  EXPECT_TRUE(r != nullptr);
  // Two components expected (neither straddles x_min).
  EXPECT_TRUE(r->components.size() == 2);
  Range::enableDonut = false;
}

void testUnionPreservesDonut() {
  section("getUnionRange: concatenation respects canonicalisation");
  Range::enableDonut = true;
  auto a = classic(-1.0, -0.5);
  auto b = classic(0.5, 1.0);
  auto r = taffo::getUnionRange(a, b);
  EXPECT_TRUE(r->components.size() == 2);
  EXPECT_NEAR(r->min, -1.0, 1e-12);
  EXPECT_NEAR(r->max, 1.0, 1e-12);
  Range::enableDonut = false;
}

int runAll() {
  const bool savedFlag = Range::enableDonut;

  testClassicAddUnchanged();
  testClassicSubUnchanged();
  testClassicMulUnchanged();

  testDonutAdd();
  testClassicPlusDonut();

  testDonutMulPositiveActivation();
  testDonutSelfSquareTight();

  testDonutDivByZeroExcludingComponents();
  testClassicDivStillBlows();
  testDonutDivSubjectCrossesZero();

  testGELUCertifiedMinimumStraddling();
  testGELUMonotoneDecreasingBranch();
  testSiLUCertifiedMinimumStraddling();
  testSiLUMonotoneIncreasingBranch();
  testGELUDonutPreservesHole();

  testUnionPreservesDonut();

  Range::enableDonut = savedFlag;
  std::fprintf(stdout, "\nDonut arithmetic test: %d passed, %d failed.\n",
               g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}

}  // namespace

int main() { return runAll(); }
