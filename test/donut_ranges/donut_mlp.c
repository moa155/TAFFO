/// TAFFO_TEST_ARGS - lm
//
// Minimal neural-network end-to-end benchmark for the CTO 2025/26
// donut-ranges + activation-models project. This is a toy 1-hidden-
// layer multi-layer perceptron with fixed, bimodal, compile-time
// weights and a ReLU activation. It is deliberately small (input
// dim = 4, hidden = 8, output = 1) so that the full analysis-to-
// fixed-point conversion pipeline completes in under a second, while
// still hitting every code path that matters:
//
//   * Weight ranges seeded via the new `scalar(range_union(...))`
//     annotation, so VRA sees explicit donut components on the
//     weights without having to infer them from the array literal.
//   * Hidden pre-activations formed by sums of (input · weight)
//     products, hitting `handleMul` and `handleAdd` with one donut
//     operand.
//   * Hidden activations produced by `taffo_relu`, exercising the
//     donut-aware `handleCallToReLU` path.
//   * Output formed by another weighted sum, with a ReLU on the
//     final output too.
//
// Compile twice: once without -vra-donut-ranges (classic) and once
// with (donut), then diff the taffo_info_vra.json and the DTA-chosen
// fixed-point formats for the hidden pre-activations. The donut build
// should allocate strictly more fractional bits than the classic
// build for at least one of the hidden pre-activations, because the
// bimodal structure of the weight array tightens the analysed range.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM     4
#define HIDDEN_DIM 8
#define OUT_DIM    1

// Bimodal weight matrix W1 (hidden x in). Values cluster around
// ±0.5 with a clear zero-excluding hole in [-0.2, 0.2].
double __attribute__((annotate(
    "scalar(range_union((-0.8, -0.2), (0.2, 0.8))) target('W1')")))
    W1[HIDDEN_DIM * IN_DIM] = {
  -0.8,  0.4, -0.3,  0.7,
   0.6, -0.5,  0.8, -0.2,
  -0.4,  0.7, -0.6,  0.3,
   0.5, -0.8,  0.2, -0.6,
  -0.7,  0.3,  0.8, -0.4,
   0.4, -0.6,  0.5,  0.7,
  -0.3,  0.8, -0.7,  0.5,
   0.7, -0.4, -0.5,  0.2,
};

// Biases live on a narrow unimodal range — no donut annotation
// needed, the classic seed is tight enough.
double __attribute__((annotate("scalar(range(-0.5, 0.5)) target('b1')")))
    b1[HIDDEN_DIM] = {
  -0.1,  0.2, -0.3,  0.1,  0.05, -0.25,  0.15, -0.05
};

// Second layer weight vector W2 (out=1 x hidden). Same bimodal
// shape as W1 but narrower hull.
double __attribute__((annotate(
    "scalar(range_union((-0.6, -0.15), (0.15, 0.6))) target('W2')")))
    W2[HIDDEN_DIM] = {
   0.5, -0.4,  0.3, -0.6,  0.4, -0.3,  0.5, -0.2
};

// NOTE: b2 is a scalar, not a 1-element array, because TAFFO's
// Conversion pass currently emits an illegal aggregate-typed `ashr`
// instruction when an annotated 1-element array is loaded whole
// (rather than through a GEP). Using the scalar avoids the corner case
// entirely; a single-output MLP does not need an array-valued bias.
double __attribute__((annotate("scalar(range(-0.3, 0.3)) target('b2')")))
    b2 = 0.05;

// Relay through a runtime barrier to stop the optimiser from folding
// the whole network to a constant.
static double pass_through(double v) {
  asm volatile("" : "+r,m"(v) : : "memory");
  return v;
}

// TAFFO recognises a call named "taffo_relu" as a ReLU activation,
// which the donut-aware handleCallToReLU processes per-component.
static double taffo_relu(double x) {
  return x > 0.0 ? x : 0.0;
}

double donut_mlp_kernel(
    double x0, double x1, double x2, double x3) {
  // Input vector is a unimodal [-1, 1] range — typical NN inputs.
  double __attribute__((annotate("scalar(range(-1.0, 1.0)) target('x')")))
      x[IN_DIM] = {
          pass_through(x0), pass_through(x1),
          pass_through(x2), pass_through(x3)
      };

  // Hidden pre-activations. Each is a sum of IN_DIM products of a
  // bimodal weight with a unimodal input, plus a small bias.
  double __attribute__((annotate("scalar() target('h_pre')")))
      h_pre[HIDDEN_DIM];
  for (int j = 0; j < HIDDEN_DIM; ++j) {
    double acc = b1[j];
    for (int i = 0; i < IN_DIM; ++i) {
      acc += W1[j * IN_DIM + i] * x[i];
    }
    h_pre[j] = acc;
  }

  // Hidden activations: ReLU over h_pre. The donut-aware
  // handleCallToReLU applied to a hidden pre-activation whose range
  // spans [-something, +something] should produce [0, +something].
  double __attribute__((annotate("scalar() target('h_post')")))
      h_post[HIDDEN_DIM];
  for (int j = 0; j < HIDDEN_DIM; ++j) {
    h_post[j] = taffo_relu(h_pre[j]);
  }

  // Output layer — one more weighted sum, then ReLU.
  double __attribute__((annotate("scalar() target('y_pre')")))
      y_pre = b2;
  for (int j = 0; j < HIDDEN_DIM; ++j) {
    y_pre += W2[j] * h_post[j];
  }
  double __attribute__((annotate("scalar() target('y_post')")))
      y_post = taffo_relu(y_pre);
  return y_post;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  double acc = 0.0;
  // A handful of runtime-looking inputs so the optimiser cannot
  // collapse everything to a constant.
  const double samples[8][IN_DIM] = {
    { 0.1,  0.2, -0.3,  0.4},
    {-0.5,  0.6,  0.7, -0.8},
    { 0.9, -1.0,  0.1,  0.2},
    { 0.3, -0.4,  0.5, -0.6},
    { 0.0,  0.0,  0.5, -0.5},
    {-0.2,  0.3,  0.0,  0.9},
    { 0.8,  0.1, -0.6,  0.4},
    {-0.9,  0.5, -0.2,  0.1},
  };
  for (int i = 0; i < 8; ++i) {
    acc += donut_mlp_kernel(samples[i][0], samples[i][1],
                            samples[i][2], samples[i][3]);
  }
  printf("mlp acc = %f\n", acc);
  return 0;
}
