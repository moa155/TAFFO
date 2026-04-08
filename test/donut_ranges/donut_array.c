/// TAFFO_TEST_ARGS - lm
//
// End-to-end demonstration of donut-range VRA on a bimodal global
// array. This is the primary end-to-end benchmark for sub-project #1
// "Donut Ranges in VRA".
//
// The computational core is 1.0 / weights[i], where `weights` is a
// compile-time constant array whose values are clustered around
// ±0.5 with strictly nothing near zero. Compile twice with the TAFFO
// driver:
//
//   # baseline: classic TAFFO VRA
//   install/bin/taffo -O2 -c -emit-llvm \
//       -o donut_array_classic.ll \
//       -temp-dir /tmp/donut_classic \
//       test/donut_ranges/donut_array.c
//
//   # donut-range VRA
//   install/bin/taffo -O2 -c -emit-llvm \
//       -Xvra -vra-donut-ranges \
//       -o donut_array_donut.ll \
//       -temp-dir /tmp/donut_donut \
//       test/donut_ranges/donut_array.c
//
// The interesting artefact is `taffo_info_vra.json` in each temp dir.
// Look for the `weights` global: in the donut build it should have
// a `components` field listing the two clusters [-0.8,-0.2] and
// [0.2, 0.8], while the classic build reports only the convex hull
// [-0.8, 0.8]. Then look for the `fdiv` instruction in
// `donut_array_kernel`: in the donut build its range is bounded by 5
// in absolute value, while the classic build reports the DIV_EPS
// blow-up [-1e8, 1e8]. DTA sees the tight donut range and allocates
// only a handful of integer bits for `y` in the donut build, against
// 28 signed integer bits in the classic build.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Compile-time constant bimodal array. Every entry is either in
// [-0.8, -0.2] or in [0.2, 0.8] — nothing near zero.
static const double weights[14] = {
  -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
   0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8
};

// Runtime index to prevent the optimiser from constant-folding the
// division into every single weight value at compile time.
static int deconstify_int(int v) {
  asm volatile("" : : "r,m"(v) : "memory");
  return v;
}

double donut_array_kernel(int i) {
  // `w` is deliberately NOT annotated with a range. That way TAFFO's
  // Initializer does not seed the phi/load with a wide convex-hull
  // annotation that would swallow the donut refinement. The range of
  // `w` is inferred by VRA from the load of `weights[i]` — which,
  // with donut-range VRA, keeps the two-component shape of the
  // underlying array.
  const int ii = deconstify_int(i);
  double w = weights[ii];

  // Annotated output: the reciprocal. Without donut ranges its
  // analysed range explodes via DIV_EPS; with donut ranges the
  // zero-exclusion fast path in handleDiv keeps it bounded by 5.
  double __attribute__((annotate("scalar() target('reciprocal')"))) y = 1.0 / w;
  return y;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  double acc = 0.0;
  for (int i = 0; i < 14; ++i) {
    acc += donut_array_kernel(i);
  }
  printf("acc = %f\n", acc);
  return 0;
}
