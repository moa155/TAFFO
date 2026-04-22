/// TAFFO_TEST_ARGS - lm
//
// End-to-end smoke test for the `scalar(range_union(...))` annotation
// syntax added to TAFFO's Initializer for the CTO 2025/26 project
// (sub-projects #1 + #3). The syntax lets the user declare, at source
// level, that a scalar value is a donut range (a union of disjoint
// closed intervals) rather than a single classic interval. The
// Initializer pass parses the syntax, populates the seeded Range's
// `components` vector, and downstream VRA uses the per-component
// information when -vra-donut-ranges is on.
//
// Usage:
//
//   install/bin/taffo -O2 -c -emit-llvm \
//       -Xvra -vra-donut-ranges \
//       -temp-dir /tmp/donut_annot \
//       -o /tmp/donut_annot/donut_annotated.ll \
//       test/donut_ranges/donut_annotated.c
//
// Inspect /tmp/donut_annot/taffo_info_vra.json: the seeded global
// `bimodal_w` should carry the exact components we declared, and the
// annotated `w_squared = w * w` should collapse to the TIGHT single
// interval [0.04, 0.64] via the self-square fast path, while the
// annotated `w_product = w * w_other` (two independent reads of the
// same donut global) should preserve the two-component donut
// [-0.64, -0.04] U [0.04, 0.64] through handleMul. Classic VRA on
// the same program yields the looser hull [-0.64, 0.64] for both.

#include <stdio.h>
#include <stdlib.h>

// A global seeded via the new annotation syntax: two disjoint
// clusters with a 0.4-wide hole around zero. With -vra-donut-ranges
// the seed carries the component list into VRA; without the flag the
// same annotation widens to the convex hull [-0.8, 0.8] and behaves
// identically to `scalar(range(-0.8, 0.8))`.
double __attribute__((annotate(
    "scalar(range_union((-0.8, -0.2), (0.2, 0.8))) target('bimodal_w')")))
    bimodal_w = 0.5;

double __attribute__((annotate(
    "scalar(range_union((-0.8, -0.2), (0.2, 0.8))) target('bimodal_w2')")))
    bimodal_w2 = 0.3;

double donut_annotated_kernel(void) {
  double w = bimodal_w;
  double w_other = bimodal_w2;

  // Square of the bimodal value. The donut-aware handleMul self-square
  // fast path collapses the two-component donut to the tight single
  // interval [0.04, 0.64]. Classic VRA would say [-0.64, 0.64].
  double __attribute__((annotate(
      "scalar() target('w_squared')"))) w2 = w * w;

  // Product of two independent donut reads. Donut-aware handleMul
  // Cartesian-products the two component lists and canonicalises:
  //   [-0.8,-0.2] * [-0.8,-0.2] = [0.04, 0.64]
  //   [-0.8,-0.2] * [ 0.2, 0.8] = [-0.64, -0.04]
  //   [ 0.2, 0.8] * [-0.8,-0.2] = [-0.64, -0.04]
  //   [ 0.2, 0.8] * [ 0.2, 0.8] = [0.04, 0.64]
  // which collapses to [-0.64,-0.04] U [0.04, 0.64] — the zero hole
  // is preserved. Classic VRA would say the single interval
  // [-0.64, 0.64].
  double __attribute__((annotate(
      "scalar() target('w_product')"))) w_prod = w * w_other;

  return w2 + w_prod;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  double acc = 0.0;
  for (int i = 0; i < 8; ++i) {
    acc += donut_annotated_kernel();
  }
  printf("acc = %f\n", acc);
  return 0;
}
