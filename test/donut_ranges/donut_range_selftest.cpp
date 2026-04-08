// Standalone self-test for donut ranges.
//
// This file deliberately avoids the gtest infrastructure because the
// existing unittests/ target references stale headers (TaffoVRA/Range.hpp,
// range_ptr_t, etc.) that no longer exist in the current Range type. Rather
// than fight that, we exercise the new donut-range code via plain C++
// assertions and print a small progress report. The supervisor can run the
// binary directly after a standard TAFFO build.
//
// Build:
//   cmake -B build -S . -DTAFFO_BUILD_TESTS=ON
//   cmake --build build --target donut_range_selftest
//   ./build/test/donut_ranges/donut_range_selftest
//
// The test only depends on the TaffoCommon library (which provides
// taffo::Range) and is linked via test/donut_ranges/CMakeLists.txt.

#include "TaffoInfo/RangeInfo.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
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

// ===== Canonicalisation =====

void testCanonicalizeSortAndMerge() {
  section("canonicalize: sort + merge touching");
  Range::enableDonut = true;

  Range r(0.0, 0.0);
  // Deliberately unsorted, with two touching components that must merge.
  r.components = {{3.0, 4.0}, {-1.0, 0.5}, {0.5, 1.0}};
  r.canonicalize();

  // {-1.0, 0.5} and {0.5, 1.0} touch -> merge into {-1.0, 1.0}.
  // Then {-1.0, 1.0} and {3.0, 4.0} stay separate -> donut of 2.
  EXPECT_TRUE(r.components.size() == 2);
  EXPECT_NEAR(r.components[0].first, -1.0, 1e-12);
  EXPECT_NEAR(r.components[0].second, 1.0, 1e-12);
  EXPECT_NEAR(r.components[1].first, 3.0, 1e-12);
  EXPECT_NEAR(r.components[1].second, 4.0, 1e-12);
  EXPECT_NEAR(r.min, -1.0, 1e-12);
  EXPECT_NEAR(r.max, 4.0, 1e-12);
}

void testCanonicalizeOverlap() {
  section("canonicalize: overlapping components merged");
  Range::enableDonut = true;

  Range r(0.0, 0.0);
  r.components = {{-2.0, 1.0}, {0.0, 3.0}};  // Overlap
  r.canonicalize();

  EXPECT_TRUE(r.components.size() == 1 ? true : r.components.size() == 0);
  // After size==1 collapse, components is cleared (classic interval semantics).
  EXPECT_TRUE(r.components.empty());
  EXPECT_NEAR(r.min, -2.0, 1e-12);
  EXPECT_NEAR(r.max, 3.0, 1e-12);
}

void testCanonicalizeSingletonCollapse() {
  section("canonicalize: 1-component donut collapses to classic interval");
  Range::enableDonut = true;

  Range r(0.0, 0.0);
  r.components = {{1.5, 2.5}};
  r.canonicalize();

  EXPECT_TRUE(r.components.empty());  // Collapsed.
  EXPECT_NEAR(r.min, 1.5, 1e-12);
  EXPECT_NEAR(r.max, 2.5, 1e-12);
}

void testCanonicalizeWidening() {
  section("canonicalize: widens past kMaxComponents by smallest-gap merge");
  Range::enableDonut = true;

  Range r(0.0, 0.0);
  // 6 components; kMaxComponents == 4. The smallest gap is between
  // {5.0, 5.1} and {5.2, 5.3} (gap 0.1), so those should be merged first.
  // The next smallest gap is between {2.0, 2.1} and {3.0, 3.1} (gap 0.9).
  r.components = {{0.0, 0.1}, {1.0, 1.1}, {2.0, 2.1},
                  {3.0, 3.1}, {5.0, 5.1}, {5.2, 5.3}};
  r.canonicalize();

  EXPECT_TRUE(r.components.size() == Range::kMaxComponents);
  // Last component must now span {5.0, 5.3} after the smallest-gap merge.
  EXPECT_NEAR(r.components.back().first, 5.0, 1e-12);
  EXPECT_NEAR(r.components.back().second, 5.3, 1e-12);
}

// ===== Serialisation round-trip =====

void testSerializeRoundTrip() {
  section("serialize/deserialize: donut components survive");
  Range::enableDonut = true;

  Range src(0.0, 0.0);
  src.components = {{-0.5, -0.1}, {0.1, 0.4}};
  src.canonicalize();

  const auto j = src.serialize();
  Range dst;
  dst.deserialize(j);

  EXPECT_TRUE(dst.components.size() == 2);
  EXPECT_NEAR(dst.components[0].first, -0.5, 1e-12);
  EXPECT_NEAR(dst.components[0].second, -0.1, 1e-12);
  EXPECT_NEAR(dst.components[1].first, 0.1, 1e-12);
  EXPECT_NEAR(dst.components[1].second, 0.4, 1e-12);
  EXPECT_NEAR(dst.min, -0.5, 1e-12);
  EXPECT_NEAR(dst.max, 0.4, 1e-12);
}

void testSerializeClassicRangeHasNoComponentsKey() {
  section("serialize: classic range emits no `components` key");
  Range::enableDonut = false;

  Range src(2.0, 5.0);
  const auto j = src.serialize();
  EXPECT_TRUE(!j.contains("components"));
}

// ===== Flag-off safety =====

void testFlagOffPreservesClassicHull() {
  section("enableDonut=false: addComponent silently widens hull");
  Range::enableDonut = false;

  Range r(1.0, 2.0);
  // With the flag off, addComponent must NOT populate `components`; it
  // should simply widen min/max so downstream callers still see a sound
  // over-approximation.
  r.addComponent(10.0, 20.0);
  EXPECT_TRUE(r.components.empty());
  EXPECT_NEAR(r.min, 1.0, 1e-12);
  EXPECT_NEAR(r.max, 20.0, 1e-12);
}

int runAll() {
  // Tests mutate Range::enableDonut; we restore the default at the end.
  const bool savedFlag = Range::enableDonut;

  testCanonicalizeSortAndMerge();
  testCanonicalizeOverlap();
  testCanonicalizeSingletonCollapse();
  testCanonicalizeWidening();
  testSerializeRoundTrip();
  testSerializeClassicRangeHasNoComponentsKey();
  testFlagOffPreservesClassicHull();

  Range::enableDonut = savedFlag;

  std::fprintf(stdout, "\nDonut range self-test: %d passed, %d failed.\n",
               g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}

}  // namespace

int main() { return runAll(); }
