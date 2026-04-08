#include "RangeOperations.hpp"
#include "RangeOperationsCallWhitelist.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <list>
#include <string>

#define DEBUG_TYPE "taffo-vra"

using namespace taffo;

#define PI 0x1.921FB54442D18p+1
#define PIO2 0x1.921FB54442D18p+0

static std::shared_ptr<Range> handleCallToCeil(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function ceil");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(ceil(op->min), ceil(op->max));
}

static std::shared_ptr<Range> handleCallToFloor(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function floor");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(floor(op->min), floor(op->max));
}

static std::shared_ptr<Range> handleCallToFabs(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function fabs");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  double min = fabs(op->min);
  double max = fabs(op->max);
  if (min <= max)
    return std::make_shared<Range>(min, max);
  return std::make_shared<Range>(max, min);
}

static std::shared_ptr<Range> handleCallToLog(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Log");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  if (op->max < 0.0)
    return std::make_shared<Range>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
  double min = op->min < 0 ? std::numeric_limits<double>::epsilon() : op->min;
  min = log(min);
  double max = log(op->max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range> handleCallToLog10(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Log10");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->max >= 0);
  double min = op->min < 0 ? std::numeric_limits<double>::epsilon() : op->min;
  min = log10(min);
  double max = log10(op->max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range> handleCallToLog2f(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Log2f");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->max >= 0);
  double min = op->min < 0 ? std::numeric_limits<double>::epsilon() : op->min;
  min = static_cast<double>(log2f(min));
  double max = log2f(op->max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range> handleCallToSqrt(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Sqrt");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->max >= 0);
  double min = op->min < 0 ? 0 : op->min;
  min = sqrt(min);
  double max = sqrt(op->max);
  if (min <= max)
    return std::make_shared<Range>(min, max);
  return std::make_shared<Range>(max, min);
}

static std::shared_ptr<Range> handleCallToExp(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Exp");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  double min = exp(op->min);
  double max = exp(op->max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range> handleCallToSin(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Sin");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // TODO: better range reduction
  if (op->min >= -PIO2 && op->max <= PIO2)
    return std::make_shared<Range>(std::sin(op->min), std::sin(op->max));

  return std::make_shared<Range>(-1.0, 1.0);
}

static std::shared_ptr<Range> handleCallToCos(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function Cos");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // TODO: better range reduction
  if (op->min >= -PI && op->max <= 0.0)
    return std::make_shared<Range>(std::cos(op->min), std::cos(op->max));
  if (op->min >= 0.0 && op->max <= PI)
    return std::make_shared<Range>(std::cos(op->max), std::cos(op->min));

  return std::make_shared<Range>(-1.0, 1.0);
}

static std::shared_ptr<Range> handleCallToAcos(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function acos");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::acos(std::max(op->min, -1.0)), std::acos(std::min(op->max, 1.0)));
}

static std::shared_ptr<Range> handleCallToAsin(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function asin");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::asin(std::max(op->min, -1.0)), std::asin(std::min(op->max, 1.0)));
}

static std::shared_ptr<Range> handleCallToAtan(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function atan");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::atan(op->min), std::atan(op->max));
}

static std::shared_ptr<Range> handleCallToTanh(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function tanh");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  /* tanh is a monotonic increasing function, output in (-1, 1). */
  return std::make_shared<Range>(std::tanh(op->min), std::tanh(op->max));
}

//-----------------------------------------------------------------------------
// Neural-network activation functions
//
// All handlers below follow the "Range Models for NN Activation Functions"
// sub-project: given an input range [lo, hi], we derive the tightest output
// range we can prove.
//
// For MONOTONIC activations (sigmoid, ReLU, leaky ReLU, ELU, softplus) the
// output range is simply [f(lo), f(hi)]; no swap is needed.
//
// For NON-MONOTONIC activations (GELU, SiLU) we perform a closed-form
// case-split around the function's known global minimum, instead of the
// earlier prototype code that scanned 1000 samples and could miss the true
// minimum if it sat between two samples.
//
// Constants below are chosen to be SOUND over-approximations of the true
// minimum value: the reported output min is strictly less than or equal to
// the exact minimum, so any downstream user sees a sound lower bound.
//
// Donut interop (sub-project #1 coupling):
// Every activation handler funnels through `applyPerComponent`, which
// iterates over the input range's components and applies a per-interval
// kernel to each one, then canonicalises the union. When the input is a
// classic single-interval range (or when donut tracking is off), the
// helper degenerates to a single kernel call — zero overhead.
//-----------------------------------------------------------------------------

/// Apply a per-interval kernel to every component of `op`, union the
/// results, and return a donut-aware Range. `kernel` receives a (lo, hi)
/// pair and returns a (lo', hi') pair describing the image of that
/// sub-interval under the activation function.
template <typename Kernel>
static std::shared_ptr<Range>
applyPerComponent(const std::shared_ptr<Range>& op, Kernel kernel) {
  const bool donut = Range::enableDonut && !op->components.empty();
  if (!donut) {
    const auto [lo, hi] = kernel(op->min, op->max);
    double a = lo, b = hi;
    if (a > b) std::swap(a, b);
    return std::make_shared<Range>(a, b);
  }

  auto result = std::make_shared<Range>(
      +std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity());
  result->components.reserve(op->components.size());
  for (const auto& c : op->components) {
    auto [lo, hi] = kernel(c.first, c.second);
    if (lo > hi) std::swap(lo, hi);
    result->components.emplace_back(lo, hi);
  }
  result->canonicalize();
  return result;
}

/// Monotonic-increasing convenience: builds a kernel (lo, hi) -> (f(lo), f(hi))
/// for `applyPerComponent` without the caller having to spell out the pair.
template <typename F>
static std::shared_ptr<Range>
applyMonotonic(const std::shared_ptr<Range>& op, F f) {
  return applyPerComponent(op, [f](double lo, double hi) {
    double rmin = f(lo);
    double rmax = f(hi);
    if (rmin > rmax) std::swap(rmin, rmax);
    return std::make_pair(rmin, rmax);
  });
}

static std::shared_ptr<Range> handleCallToSigmoid(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function sigmoid");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  // sigmoid(x) = 1 / (1 + exp(-x)), monotonic increasing, output in (0, 1).
  return applyMonotonic(op, [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

static std::shared_ptr<Range> handleCallToReLU(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function relu");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  // relu(x) = max(0, x), piecewise monotonic non-decreasing, output in [0, +inf).
  return applyMonotonic(op, [](double x) { return std::max(0.0, x); });
}

static std::shared_ptr<Range> handleCallToLeakyReLU(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function leaky_relu");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  // leaky_relu(x) = x if x >= 0 else alpha*x. With the usual alpha = 0.01 > 0
  // this is strictly monotonic increasing everywhere.
  constexpr double kAlpha = 0.01;
  return applyMonotonic(op, [](double x) { return x >= 0.0 ? x : kAlpha * x; });
}

static std::shared_ptr<Range> handleCallToELU(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function elu");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  // elu(x) = x if x >= 0 else (exp(x) - 1). Monotonic increasing, output in (-1, +inf).
  return applyMonotonic(op, [](double x) { return x >= 0.0 ? x : std::exp(x) - 1.0; });
}

static std::shared_ptr<Range> handleCallToSoftplus(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function softplus");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  // softplus(x) = log(1 + exp(x)), monotonic increasing, output in (0, +inf).
  // Uses log1p/exp to stay numerically stable for large negative x where
  // exp(x) underflows but log(1 + exp(x)) is still meaningful.
  return applyMonotonic(op, [](double x) {
    if (x > 30.0) return x;                // exp overflows; softplus(x) ~ x
    if (x < -30.0) return 0.0;             // exp(x) underflows to 0
    return std::log1p(std::exp(x));
  });
}

/// Build a kernel that performs closed-form range analysis for a
/// NON-MONOTONIC activation with a single global minimum.
///
/// The function is assumed to be:
///   * monotonic decreasing on (-inf, X_MIN]
///   * monotonic increasing on [X_MIN, +inf)
///   * bounded below by Y_MIN at X_MIN (Y_MIN is a sound under-approximation
///     of the true minimum value, so any hull built from it is sound).
///
/// X_MIN_HI is the left edge of the interval guaranteed to contain the true
/// minimum; X_MIN is the right edge. Together they sandwich the true x_min
/// so case splits stay conservative.
template <typename F>
static std::pair<double, double>
nonMonotonicKernel(double lo, double hi, F f,
                   double X_MIN_HI, double X_MIN, double Y_MIN) {
  // Case A: interval sits entirely on the monotonic-decreasing branch.
  if (hi <= X_MIN_HI) {
    return {f(hi), f(lo)};
  }
  // Case B: interval sits entirely on the monotonic-increasing branch.
  if (lo >= X_MIN) {
    return {f(lo), f(hi)};
  }
  // Case C: interval straddles the minimum. Output min is Y_MIN; output
  // max is whichever endpoint is further from X_MIN.
  return {Y_MIN, std::max(f(lo), f(hi))};
}

static std::shared_ptr<Range> handleCallToGELU(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function gelu");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // GELU (tanh approximation):
  //   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  //
  // True minimum for this approximation is at x ≈ -0.7517916 with value
  // ≈ -0.1700432. The constants below sandwich that location and use a
  // slightly-more-negative Y_MIN so the reported range is a sound
  // over-approximation of the true image.
  constexpr double X_MIN = -0.75;
  constexpr double X_MIN_HI = -0.76;
  constexpr double Y_MIN = -0.1701;
  auto gelu = [](double x) {
    return 0.5 * x * (1.0 + std::tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  };
  return applyPerComponent(op, [&](double lo, double hi) {
    return nonMonotonicKernel(lo, hi, gelu, X_MIN_HI, X_MIN, Y_MIN);
  });
}

static std::shared_ptr<Range> handleCallToSiLU(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 1 && "too many operands in function silu");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // SiLU (Swish): silu(x) = x * sigmoid(x).
  // True minimum is at x ≈ -1.2784645 with value ≈ -0.2784645.
  constexpr double X_MIN = -1.27;
  constexpr double X_MIN_HI = -1.29;
  constexpr double Y_MIN = -0.2785;
  auto silu = [](double x) {
    if (x < -30.0) return 0.0;
    return x / (1.0 + std::exp(-x));
  };
  return applyPerComponent(op, [&](double lo, double hi) {
    return nonMonotonicKernel(lo, hi, silu, X_MIN_HI, X_MIN, Y_MIN);
  });
}

static std::shared_ptr<Range> handleCallToRand(const std::list<std::shared_ptr<Range>>& operands) {
  // FIXME: RAND_MAX is implementation defined!
  return std::make_shared<Range>(0, RAND_MAX);
}

static std::shared_ptr<Range> handleCallToFMA(const std::list<std::shared_ptr<Range>>& operands) {
  assert(operands.size() == 3 && "Wrong number of operands in FMA");
  std::shared_ptr<Range> op1 = operands.front();
  std::shared_ptr<Range> op2 = *(++operands.begin());
  std::shared_ptr<Range> op3 = operands.back();
  if (!op1 || !op2 || !op3)
    return nullptr;
  return handleAdd(handleMul(op1, op2), op3);
}

const std::map<const std::string, map_value_t> taffo::functionWhiteList = {
  CMATH_WHITELIST_FUN("ceil", &handleCallToCeil),
  CMATH_WHITELIST_FUN("floor", &handleCallToFloor),
  CMATH_WHITELIST_FUN("fabs", &handleCallToFabs),
  CMATH_WHITELIST_FUN("log", &handleCallToLog),
  CMATH_WHITELIST_FUN("log10", &handleCallToLog10),
  CMATH_WHITELIST_FUN("log2", &handleCallToLog2f),
  CMATH_WHITELIST_FUN("sqrt", &handleCallToSqrt),
  CMATH_WHITELIST_FUN("exp", &handleCallToExp),
  CMATH_WHITELIST_FUN("sin", &handleCallToSin),
  CMATH_WHITELIST_FUN("cos", &handleCallToCos),
  CMATH_WHITELIST_FUN("acos", &handleCallToAcos),
  CMATH_WHITELIST_FUN("asin", &handleCallToAsin),
  CMATH_WHITELIST_FUN("atan", &handleCallToAtan),
  CMATH_WHITELIST_FUN("tanh", &handleCallToTanh),
  CMATH_WHITELIST_FUN("rand", &handleCallToRand),
  CMATH_WHITELIST_FUN("fma", &handleCallToFMA),
  CMATH_WHITELIST_FUN("sigmoid", &handleCallToSigmoid),
  CMATH_WHITELIST_FUN("relu", &handleCallToReLU),
  CMATH_WHITELIST_FUN("leaky_relu", &handleCallToLeakyReLU),
  CMATH_WHITELIST_FUN("elu", &handleCallToELU),
  CMATH_WHITELIST_FUN("softplus", &handleCallToSoftplus),
  CMATH_WHITELIST_FUN("gelu", &handleCallToGELU),
  CMATH_WHITELIST_FUN("silu", &handleCallToSiLU),
  CMATH_WHITELIST_FUN("swish", &handleCallToSiLU),  // PyTorch/TF alias for SiLU
  INTRINSIC_WHITELIST_FUN("fmuladd", &handleCallToFMA)};
