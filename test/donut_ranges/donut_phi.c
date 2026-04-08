/// TAFFO_TEST_ARGS - lm
//
// End-to-end demonstration of donut-range VRA on a phi-node pattern.
//
// This file is meant to be compiled twice by the full TAFFO driver:
//
//   # baseline: classic TAFFO VRA
//   taffo -O2 -o donut_phi_classic  test/donut_ranges/donut_phi.c
//
//   # donut-range VRA
//   taffo -O2 -mllvm -vra-donut-ranges \
//         -o donut_phi_donut    test/donut_ranges/donut_phi.c
//
// The computational core is a reciprocal applied to a value that can
// take one of two disjoint ranges selected by a runtime flag:
//
//                [-0.8, -0.2]            if pick_neg
//        w  =  {                                                   (*)
//                [ 0.2,  0.8]            otherwise
//
// In classic interval VRA the phi node merging the two branches
// collapses to the convex hull [-0.8, 0.8], which *crosses zero*.
// TAFFO's handleDiv then has to apply its `DIV_EPS = 1e-8` nudge to
// avoid a literal division by zero, and the reciprocal of w explodes
// to roughly [-1e8, 1e8]. DTA is then forced to allocate 27-28
// signed integer bits for y, a variable whose concrete magnitude is
// provably bounded by 5.
//
// With -vra-donut-ranges the phi node uses the donut-aware
// getUnionRange (see lib/TaffoVRA/TaffoVRA/RangeOperations.cpp
// §handleUnionRange). Because the two incoming values are disjoint,
// the result is a genuine two-component donut
//     [-0.8, -0.2] U [0.2, 0.8]
// and handleDiv's zero-exclusion fast path kicks in, giving the
// tight output
//     [-5, -1.25] U [1.25, 5]
// DTA then needs only 4 signed integer bits for y — a 24-bit saving.
//
// The interesting output is NOT the numerical value of y at runtime
// (which the annotation pins inside the same set in both builds);
// it is the `newType` assigned to y in `taffo_info.json` at the end
// of the VRA pass, which directly reflects the bit-width savings.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Prevent the optimiser from folding the reciprocal at compile time.
static double deconstify(double v) {
  asm volatile("" : : "r,m"(v) : "memory");
  return v;
}

double donut_phi_kernel(int pick_neg) {
  // NOTE: we deliberately do NOT annotate the outer `w` with a wide
  // convex-hull range such as `scalar(range(-0.8, 0.8))`. If we did,
  // VRA's `handlePhiNode` would seed the phi's starting range with
  // that classic interval, and the narrower disjoint ranges of the
  // two branch temporaries (below) would be SWALLOWED by it during
  // canonicalisation — the donut hole would be lost.
  //
  // Instead, we annotate the per-branch temporaries with their own
  // narrow, strictly disjoint ranges. VRA sees `wneg ∈ [-0.8, -0.2]`
  // and `wpos ∈ [0.2, 0.8]` and, at the merge point, computes
  //     phi(w) = union(wneg, wpos)
  // With donut tracking on, the union is a two-component donut; with
  // donut tracking off, it collapses to [-0.8, 0.8] — exactly the
  // comparison this benchmark is here to demonstrate.
  double w;

  if (pick_neg) {
    double __attribute__((annotate("scalar(range(-0.8, -0.2))"))) wneg = deconstify(-0.5);
    w = wneg;
  } else {
    double __attribute__((annotate("scalar(range(0.2, 0.8))"))) wpos = deconstify( 0.5);
    w = wpos;
  }

  // y is the reciprocal of the (classic-or-donut) w. Without donut
  // ranges this division hits the DIV_EPS nudge and y's range blows
  // up to roughly [-1e8, 1e8]. With donut ranges the zero-exclusion
  // fast path kicks in and y stays bounded by 5 in absolute value.
  double __attribute__((annotate("scalar() target('reciprocal')"))) y = 1.0 / w;

  return y;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // Exercise both branches so the phi node is actually emitted and
  // both incoming values are kept alive by TAFFO's dead-code pass.
  double acc = 0.0;
  for (int i = 0; i < 4; ++i) {
    acc += donut_phi_kernel(i & 1);
  }
  printf("acc = %f\n", acc);
  return 0;
}
