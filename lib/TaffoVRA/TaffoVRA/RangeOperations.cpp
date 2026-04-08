#include "RangeOperations.hpp"
#include "RangeOperationsCallWhitelist.hpp"
#include "Utils/PtrCasts.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <assert.h>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;

//=============================================================================
// Donut-range helpers
//
// A donut range is a Range whose `components` vector is non-empty, meaning
// the value is constrained to a finite union of disjoint closed intervals
// rather than a single convex hull [min, max].
//
// These helpers implement "piecewise" binary arithmetic: given two operands,
// walk the Cartesian product of their components (or their single-interval
// hull when they are classic ranges), apply a user-supplied per-interval
// operator, and collect every resulting sub-interval into a canonical
// donut. When the feature flag `Range::enableDonut` is OFF, the helpers
// fall back to the classic convex-hull arithmetic so there is zero
// behavioural change for existing TAFFO users.
//=============================================================================

using IntervalOp =
    std::function<std::pair<double, double>(double, double, double, double)>;

/// Runs `op` on each pair of components drawn from lhs x rhs and collects
/// the resulting intervals into a new donut-aware Range. Guaranteed to
/// short-circuit to a classic single-interval result when neither operand
/// is a donut (or when the donut flag is off).
static std::shared_ptr<Range>
applyPiecewise(const std::shared_ptr<Range>& lhs,
               const std::shared_ptr<Range>& rhs,
               IntervalOp op) {
  if (!lhs || !rhs) return nullptr;

  // Fast path: both operands are classic intervals or the flag is off.
  // In either case we only need to apply `op` once to the hulls, which
  // exactly reproduces the pre-donut behaviour.
  const bool useDonut = Range::enableDonut && (!lhs->components.empty() ||
                                                !rhs->components.empty());
  if (!useDonut) {
    const auto [lo, hi] = op(lhs->min, lhs->max, rhs->min, rhs->max);
    return std::make_shared<Range>(lo, hi);
  }

  const auto lComps = lhs->getComponentsOrHull();
  const auto rComps = rhs->getComponentsOrHull();

  auto result = std::make_shared<Range>(
      +std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity());
  // We deliberately bypass addComponent's "flag off => no-op" branch by
  // assembling `components` directly here: the flag IS on in this branch.
  result->components.reserve(lComps.size() * rComps.size());
  for (const auto& l : lComps) {
    for (const auto& r : rComps) {
      const auto [lo, hi] = op(l.first, l.second, r.first, r.second);
      result->components.emplace_back(lo, hi);
    }
  }
  result->canonicalize();
  return result;
}

//-----------------------------------------------------------------------------
// Wrappers
//-----------------------------------------------------------------------------

/** Handle binary instructions */
std::shared_ptr<Range> taffo::handleBinaryInstruction(const std::shared_ptr<Range> op1,
                                                      const std::shared_ptr<Range> op2,
                                                      const unsigned OpCode) {
  switch (OpCode) {
  case Instruction::Add:
  case Instruction::FAdd: return handleAdd(op1, op2); break;
  case Instruction::Sub:
  case Instruction::FSub: return handleSub(op1, op2); break;
  case Instruction::Mul:
  case Instruction::FMul: return handleMul(op1, op2); break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv: return handleDiv(op1, op2); break;
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem: return handleRem(op1, op2); break;
  case Instruction::Shl:  return handleShl(op1, op2);
  case Instruction::LShr: // TODO implement
  case Instruction::AShr: return handleAShr(op1, op2);
  case Instruction::And:  // TODO implement
  case Instruction::Or:   // TODO implement
  case Instruction::Xor:  // TODO implement
    break;
  default:
    assert(false);        // unsupported operation
    break;
  }
  return nullptr;
}

std::shared_ptr<Range> taffo::handleUnaryInstruction(const std::shared_ptr<Range> op, const unsigned OpCode) {
  if (!op)
    return nullptr;

  switch (OpCode) {
  case Instruction::FNeg: return std::make_shared<Range>(-op->max, -op->min); break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

/** Cast instructions */
std::shared_ptr<Range>
taffo::handleCastInstruction(const std::shared_ptr<Range> scalar, const unsigned OpCode, const Type* dest) {
  switch (OpCode) {
  case Instruction::Trunc:    return handleTrunc(scalar, dest); break;
  case Instruction::ZExt:
  case Instruction::SExt:     return copyRange(scalar); break;
  case Instruction::FPToUI:   return handleCastToUI(scalar); break;
  case Instruction::FPToSI:   return handleCastToSI(scalar); break;
  case Instruction::UIToFP:
  case Instruction::SIToFP:   return copyRange(scalar); break;
  case Instruction::FPTrunc:  return handleFPTrunc(scalar, dest);
  case Instruction::FPExt:    return copyRange(scalar); break;
  case Instruction::PtrToInt:
  case Instruction::IntToPtr: return handleCastToSI(scalar); break;
  case Instruction::BitCast: // TODO check
    return copyRange(scalar);
    break;
  case Instruction::AddrSpaceCast: return copyRange(scalar); break;
  default:
    assert(false);           // unsupported operation
    break;
  }
  return nullptr;
}

/** Return true if this function call can be handled by taffo::handleMathCallInstruction */
bool taffo::isMathCallInstruction(const std::string& function) { return functionWhiteList.count(function); }

/** Handle call to known math functions. Return nullptr if unknown */
std::shared_ptr<Range> taffo::handleMathCallInstruction(const std::list<std::shared_ptr<Range>>& ops,
                                                        const std::string& function) {
  const auto it = functionWhiteList.find(function);
  if (it != functionWhiteList.end())
    return it->second(ops);
  return nullptr;
}

/** Handle call to known math functions. Return nullptr if unknown */
std::shared_ptr<Range> taffo::handleCompare(const std::list<std::shared_ptr<Range>>& ops,
                                            const CmpInst::Predicate pred) {
  switch (pred) {
  case CmpInst::Predicate::FCMP_FALSE: return getAlwaysFalse();
  case CmpInst::Predicate::FCMP_TRUE:  return getAlwaysTrue();
  default:                             break;
  }

  // from now on only 2 operators compare
  assert(ops.size() > 1 && "too few operators in compare instruction");
  assert(ops.size() <= 2 && "too many operators in compare instruction");

  // extract values for easy access
  std::shared_ptr<Range> lhs = ops.front();
  std::shared_ptr<Range> rhs = ops.back();
  // if unavailable data, nothing can be said
  if (!lhs || !rhs)
    return getGenericBoolRange();

  // NOTE: not dealing with Ordered / Unordered variants
  switch (pred) {
  case CmpInst::Predicate::FCMP_OEQ:
  case CmpInst::Predicate::FCMP_UEQ:
  case CmpInst::Predicate::ICMP_EQ:
    if (lhs->min == lhs->max && rhs->min == rhs->max && lhs->min == rhs->min)
      return getAlwaysTrue();
    else if (lhs->max < rhs->min || rhs->max < lhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OGT:
  case CmpInst::Predicate::FCMP_UGT:
  case CmpInst::Predicate::ICMP_UGT:
  case CmpInst::Predicate::ICMP_SGT:
    if (lhs->min > rhs->max)
      return getAlwaysTrue();
    else if (lhs->max <= rhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OGE:
  case CmpInst::Predicate::FCMP_UGE:
  case CmpInst::Predicate::ICMP_UGE:
  case CmpInst::Predicate::ICMP_SGE:
    if (lhs->min >= rhs->max)
      return getAlwaysTrue();
    else if (lhs->max < rhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OLT:
  case CmpInst::Predicate::FCMP_ULT:
  case CmpInst::Predicate::ICMP_ULT:
  case CmpInst::Predicate::ICMP_SLT:
    if (lhs->max < rhs->min)
      return getAlwaysTrue();
    else if (lhs->min >= rhs->max)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OLE:
  case CmpInst::Predicate::FCMP_ULE:
  case CmpInst::Predicate::ICMP_ULE:
  case CmpInst::Predicate::ICMP_SLE:
    if (lhs->max <= rhs->min)
      return getAlwaysTrue();
    else if (lhs->min > rhs->max)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_ONE:
  case CmpInst::Predicate::FCMP_UNE:
  case CmpInst::Predicate::ICMP_NE:
    if (lhs->min == lhs->max && rhs->min == rhs->max && lhs->min == rhs->min)
      return getAlwaysFalse();
    else if (lhs->max < rhs->min || rhs->max < lhs->min)
      return getAlwaysTrue();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_ORD: // none of the operands is NaN
  case CmpInst::Predicate::FCMP_UNO: // one of the operand is NaN
    // TODO implement
    break;
  default: break;
  }
  return nullptr;
}

//-----------------------------------------------------------------------------
// Arithmetic
//-----------------------------------------------------------------------------

/** operator+ */
std::shared_ptr<Range> taffo::handleAdd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  return applyPiecewise(op1, op2,
                         [](double a_lo, double a_hi, double b_lo, double b_hi) {
                           return std::make_pair(a_lo + b_lo, a_hi + b_hi);
                         });
}

/** operator- */
std::shared_ptr<Range> taffo::handleSub(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  return applyPiecewise(op1, op2,
                         [](double a_lo, double a_hi, double b_lo, double b_hi) {
                           return std::make_pair(a_lo - b_hi, a_hi - b_lo);
                         });
}

// Per-interval square: image of [lo, hi] under x -> x*x. The lower bound is
// 0 iff the interval straddles zero, and the upper bound is max(lo^2, hi^2).
// This is strictly tighter than the Cartesian product formula for generic
// multiplication, which would produce spurious negative components because
// it naively multiplies [-a,-b]*[b,a] without noticing the operands are
// the same SSA value.
static std::pair<double, double> squareInterval(double lo, double hi) {
  const double a = lo * lo;
  const double b = hi * hi;
  const double rMax = std::max(a, b);
  const double rMin = (lo <= 0.0 && hi >= 0.0) ? 0.0 : std::min(a, b);
  return {rMin, rMax};
}

/** operator*
 *
 * Two fast paths:
 *   1. Self-square (op1 == op2): use the closed-form `squareInterval`
 *      helper per component, then canonicalise. This preserves the
 *      "x*x >= 0" invariant that the generic Cartesian product would
 *      otherwise destroy — see report §7.5 for the original imprecision
 *      that motivated this branch.
 *   2. Generic multiplication: route through `applyPiecewise` with the
 *      textbook (min, max) of the four corners.
 */
std::shared_ptr<Range> taffo::handleMul(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;

  // Self-square fast path. Applies to both classic and donut operands: in
  // the classic case it reproduces the pre-project behaviour exactly; in
  // the donut case it walks the component list and squares each one
  // independently, avoiding the spurious negative sub-intervals that a
  // generic Cartesian product would generate for mirror-symmetric
  // donuts such as `[-2,-0.5] ∪ [0.5,2]`.
  if (op1 == op2) {
    const bool donut = Range::enableDonut && !op1->components.empty();
    if (!donut) {
      const auto [lo, hi] = squareInterval(op1->min, op1->max);
      return std::make_shared<Range>(lo, hi);
    }
    auto result = std::make_shared<Range>(
        +std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    result->components.reserve(op1->components.size());
    for (const auto& c : op1->components) {
      const auto [lo, hi] = squareInterval(c.first, c.second);
      result->components.emplace_back(lo, hi);
    }
    result->canonicalize();
    return result;
  }

  return applyPiecewise(op1, op2,
                         [](double a_lo, double a_hi, double b_lo, double b_hi) {
                           const double a = a_lo * b_lo;
                           const double b = a_hi * b_hi;
                           const double c = a_lo * b_hi;
                           const double d = a_hi * b_lo;
                           return std::make_pair(std::min({a, b, c, d}),
                                                 std::max({a, b, c, d}));
                         });
}

/** operator/
 *
 * When donut ranges are off, or when the divisor is a classic interval that
 * crosses zero, we keep the legacy DIV_EPS dance that nudges the divisor
 * off zero and accepts a wide output range.
 *
 * When donut ranges are on AND the divisor's components each avoid zero,
 * we get the big win: each sub-interval is strictly positive or strictly
 * negative, so we divide by it with the textbook interval formula and
 * union the results — no DIV_EPS fudge required.
 */
std::shared_ptr<Range> taffo::handleDiv(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;

  auto safeIntervalDiv = [](double a_lo, double a_hi,
                            double b_lo, double b_hi) {
#define DIV_EPS (static_cast<double>(1e-8))
    double bl = b_lo;
    double bh = b_hi;
    if (bh <= 0.0) {
      bl = std::min(bl, -DIV_EPS);
      bh = std::min(bh, -DIV_EPS);
    } else if (bl < 0.0) {
      bl = -DIV_EPS;
      bh = +DIV_EPS;
    } else {
      bl = std::max(bl, +DIV_EPS);
      bh = std::max(bh, +DIV_EPS);
    }
    const double a = a_lo / bl;
    const double b = a_hi / bh;
    const double c = a_lo / bh;
    const double d = a_hi / bl;
    return std::make_pair(std::min({a, b, c, d}), std::max({a, b, c, d}));
#undef DIV_EPS
  };

  // Donut-aware fast path: iterate over every pair of sub-intervals.
  // Each divisor component that excludes zero benefits from exact
  // interval division (no DIV_EPS nudge), while any component that still
  // crosses zero falls back to safeIntervalDiv.
  const bool useDonut =
      Range::enableDonut && (!op1->components.empty() || !op2->components.empty());
  if (!useDonut) {
    const auto [lo, hi] =
        safeIntervalDiv(op1->min, op1->max, op2->min, op2->max);
    return std::make_shared<Range>(lo, hi);
  }

  const auto lComps = op1->getComponentsOrHull();
  const auto rComps = op2->getComponentsOrHull();
  auto result = std::make_shared<Range>(
      +std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity());
  result->components.reserve(lComps.size() * rComps.size());
  for (const auto& l : lComps) {
    for (const auto& r : rComps) {
      const bool crossesZero = r.first <= 0.0 && r.second >= 0.0;
      std::pair<double, double> sub;
      if (crossesZero) {
        sub = safeIntervalDiv(l.first, l.second, r.first, r.second);
      } else {
        // Textbook interval division: no DIV_EPS because r is known
        // strictly positive or strictly negative.
        const double a = l.first / r.first;
        const double b = l.second / r.second;
        const double c = l.first / r.second;
        const double d = l.second / r.first;
        sub = {std::min({a, b, c, d}), std::max({a, b, c, d})};
      }
      result->components.emplace_back(sub);
    }
  }
  result->canonicalize();
  return result;
}

double getRemMin(double op1_min, double op1_max, double op2_min, double op2_max);
double getRemMax(double op1_min, double op1_max, double op2_min, double op2_max);

/** operator% */
std::shared_ptr<Range> taffo::handleRem(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  const double min_value = getRemMin(op1->min, op1->max, op2->min, op2->max);
  const double max_value = getRemMax(op1->min, op1->max, op2->min, op2->max);
  return std::make_shared<Range>(min_value, max_value);
}

double getRemMin(double op1_min, double op1_max, double op2_min, double op2_max) {
  // the sign of the second operand does not affect the result, we always mirror negative into positive
  if (op2_max < 0)
    return getRemMin(op1_min, op1_max, -op2_max, -op2_min);
  if (op2_min < 0) {
    // we have to split second operand range into negative and positive parts and calculate min separately
    double neg = getRemMin(op1_min, op1_max, 1, -op2_min);
    double pos = getRemMin(op1_min, op1_max, 0, op2_max);
    return std::min(neg, pos);
  }
  if (op1_min >= 0) {
    // this is the case when remainder will always return a non-negative result
    // if any of the limits are 0, the min will always be 0
    if (op1_min == 0.0 || op1_max == 0.0)
      return 0.0;
    // the intervals are intersecting, there is always going to be n % n = 0
    if (op1_max >= op2_min && op1_min <= op2_max)
      return 0.0;
    // the first argument range is strictly lower than the second,
    // the mod is always going to return values from the first interval, just take the lowest
    if (op1_max < op2_min)
      return op1_min;
    // the first range is strictly higher that the second
    // we cannot tell the exact min, so return 0 as this is the lowest it can be
    return 0.0;
  }
  else if (op1_max < 0) {
    // this is the case when % will always return negative result
    // mirror the interval into positives and calculate max with "-" sign as the minimum
    double neg = -getRemMax(-op1_max, -op1_min, op2_min, op2_max);
    return neg;
  }
  else {
    // we need to split the interval into the negative and positive parts
    // first, we take the negative part of the interval [op1_min, -1]
    // we mirror it to [1, -op1_min], which is going to be positive
    // we calculate the max and take it with the "-" sign as the minimum value
    double neg = -getRemMax(1.0, -op1_min, op2_min, op2_max);
    // for the positive part we calculate it the standard way
    double pos = getRemMin(0.0, op1_max, op2_min, op2_max);
    return std::min(neg, pos);
  }
}

double getRemMax(double op1_min, double op1_max, double op2_min, double op2_max) {
  if (op1_min >= 0) {
    // this is the case when % will always return non-negative result
    // the range might include n*op2_max+(op2_max-1) value that will be the max
    if (op1_max >= op2_max)
      return op2_max - 1;
    // op1_max < op2_max, so op1_max % op2_max = op1_max
    return op1_max;
  }
  else if (op1_max < 0) {
    // this is the case when remainder will always return a negative result, we need to choose the highest max
    // mirror the interval and calculate the min, take it with "-" sign as max
    return -getRemMin(-op1_max, -op1_min, op2_min, op2_max);
  }
  else {
    // we can ignore the negative part of the interval as it always will be lower than the positive
    return getRemMax(0.0, op1_max, op2_min, op2_max);
  }
}

std::shared_ptr<Range> taffo::handleShl(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  // FIXME: it only works if no overflow occurs.
  if (!op1 || !op2)
    return nullptr;
  const unsigned sh_min = static_cast<unsigned>(op2->min);
  const unsigned sh_max = static_cast<unsigned>(op2->max);
  const long op_min = static_cast<long>(op1->min);
  const long op_max = static_cast<long>(op1->max);
  return std::make_shared<Range>(static_cast<double>(op_min << ((op_min < 0) ? sh_max : sh_min)),
                                 static_cast<double>(op_max << ((op_max < 0) ? sh_min : sh_max)));
}

std::shared_ptr<Range> taffo::handleAShr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  const unsigned sh_min = static_cast<unsigned>(op2->min);
  const unsigned sh_max = static_cast<unsigned>(op2->max);
  const long op_min = static_cast<long>(op1->min);
  const long op_max = static_cast<long>(op1->max);
  return std::make_shared<Range>(static_cast<double>(op_min >> ((op_min > 0) ? sh_max : sh_min)),
                                 static_cast<double>(op_max >> ((op_max > 0) ? sh_min : sh_max)));
}

/** Trunc */
std::shared_ptr<Range> taffo::handleTrunc(const std::shared_ptr<Range> op, const Type* dest) {
  using namespace llvm;
  if (!op)
    return nullptr;
  const IntegerType* itype = cast<IntegerType>(dest);

  APSInt imin(64U, true), imax(64U, true);
  bool isExact;
  APFloat(op->min).convertToInteger(imin, APFloatBase::roundingMode::TowardNegative, &isExact);
  APFloat(op->max).convertToInteger(imax, APFloatBase::roundingMode::TowardPositive, &isExact);
  APSInt new_imin(imin.trunc(itype->getBitWidth()));
  APSInt new_imax(imax.trunc(itype->getBitWidth()));

  return std::make_shared<Range>(new_imin.getExtValue(), new_imax.getExtValue());
}

/** CastToUInteger */
std::shared_ptr<Range> taffo::handleCastToUI(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  const double r1 = static_cast<double>(static_cast<unsigned long>(op->min));
  const double r2 = static_cast<double>(static_cast<unsigned long>(op->max));
  return std::make_shared<Range>(r1, r2);
}

/** CastToUInteger */
std::shared_ptr<Range> taffo::handleCastToSI(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  const double r1 = static_cast<double>(static_cast<long>(op->min));
  const double r2 = static_cast<double>(static_cast<long>(op->max));
  return std::make_shared<Range>(r1, r2);
}

/** FPTrunc */
std::shared_ptr<Range> taffo::handleFPTrunc(const std::shared_ptr<Range> gop, const Type* dest) {
  if (!gop)
    return nullptr;
  assert(dest && dest->isFloatingPointTy() && "Non-floating-point destination Type.");

  APFloat apmin(gop->min);
  APFloat apmax(gop->max);
  // Convert with most conservative rounding mode
  bool losesInfo;
  apmin.convert(dest->getFltSemantics(), APFloatBase::rmTowardNegative, &losesInfo);
  apmax.convert(dest->getFltSemantics(), APFloatBase::rmTowardPositive, &losesInfo);

  // Convert back to double
  apmin.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardNegative, &losesInfo);
  apmax.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardPositive, &losesInfo);
  return std::make_shared<Range>(apmin.convertToDouble(), apmax.convertToDouble());
}

/** boolean Xor instruction */
std::shared_ptr<Range> taffo::handleBooleanXor(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() && !op2->cross())
    return getAlwaysFalse();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** boolean And instruction */
std::shared_ptr<Range> taffo::handleBooleanAnd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() && !op2->cross())
    return getAlwaysTrue();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** boolean Or instruction */
std::shared_ptr<Range> taffo::handleBooleanOr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() || !op2->cross())
    return getAlwaysTrue();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** deep copy of range */
std::shared_ptr<ValueInfoWithRange> taffo::copyRange(const std::shared_ptr<ValueInfoWithRange> op) {
  if (!op)
    return nullptr;

  if (const std::shared_ptr<ScalarInfo> op_s = std::dynamic_ptr_cast<ScalarInfo>(op))
    return std::static_ptr_cast<ValueInfoWithRange>(op_s->clone());

  const std::shared_ptr<StructInfo> op_s = std::static_ptr_cast<StructInfo>(op);
  SmallVector<std::shared_ptr<ValueInfo>, 4> new_fields;
  unsigned num_fields = op_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; i++) {
    if (std::shared_ptr<ValueInfo> field = op_s->getField(i))
      if (std::shared_ptr<PointerInfo> ptr_field = std::dynamic_ptr_cast_or_null<PointerInfo>(field))
        new_fields.push_back(std::make_shared<PointerInfo>(ptr_field->getPointed()));
      else
        new_fields.push_back(copyRange(std::static_ptr_cast<ValueInfoWithRange>(field)));
    else
      new_fields.push_back(nullptr);
  }
  return std::make_shared<StructInfo>(new_fields);
}

std::shared_ptr<Range> taffo::copyRange(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  return std::static_ptr_cast<Range>(op->clone());
}

/** create a generic boolean range */
std::shared_ptr<Range> taffo::getGenericBoolRange() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(0), static_cast<double>(1));
  return res;
}

/** create a always false boolean range */
std::shared_ptr<Range> taffo::getAlwaysFalse() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(0), static_cast<double>(0));
  return res;
}

/** create a always false boolean range */
std::shared_ptr<Range> taffo::getAlwaysTrue() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(1), static_cast<double>(1));
  return res;
}

/** create a union between ranges */
std::shared_ptr<Range> taffo::getUnionRange(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1)
    return copyRange(op2);
  if (!op2)
    return copyRange(op1);
  // Classic (and always-correct) convex-hull union: downstream callers
  // that do not know about donut ranges see this hull no matter what.
  const double min = std::min({op1->min, op2->min});
  const double max = std::max({op1->max, op2->max});
  auto result = std::make_shared<Range>(min, max);

  // Donut-aware refinement: when the flag is on, ALWAYS run the
  // component-level union (not just when the operands already carry
  // components). This is what lets two *classic* disjoint intervals
  // join into a proper two-component donut — which is the whole point
  // of sub-project #1. canonicalize() will automatically collapse back
  // to a classic interval when the two operands happen to overlap.
  if (Range::enableDonut) {
    const auto lhs = op1->getComponentsOrHull();
    const auto rhs = op2->getComponentsOrHull();
    result->components.reserve(lhs.size() + rhs.size());
    for (const auto& c : lhs) result->components.push_back(c);
    for (const auto& c : rhs) result->components.push_back(c);
    result->canonicalize();
  }
  return result;
}

std::shared_ptr<ValueInfoWithRange> taffo::getUnionRange(const std::shared_ptr<ValueInfoWithRange> op1,
                                                         const std::shared_ptr<ValueInfoWithRange> op2) {
  if (!op1)
    return copyRange(op2);
  if (!op2)
    return copyRange(op1);

  if (const std::shared_ptr<ScalarInfo> sop1 = std::dynamic_ptr_cast<ScalarInfo>(op1)) {
    const std::shared_ptr<ScalarInfo> sop2 = std::static_ptr_cast<ScalarInfo>(op2);
    return std::make_shared<ScalarInfo>(nullptr, sop1->range->join(sop2->range));
  }

  const std::shared_ptr<StructInfo> op1_s = std::static_ptr_cast<StructInfo>(op1);
  const std::shared_ptr<StructInfo> op2_s = std::static_ptr_cast<StructInfo>(op2);
  unsigned num_fields = std::max(op1_s->getNumFields(), op2_s->getNumFields());
  SmallVector<std::shared_ptr<ValueInfo>, 4U> new_fields;
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    const std::shared_ptr<ValueInfo> op1_f = op1_s->getField(i);
    if (op1_f && std::isa_ptr<PointerInfo>(op1_f)) {
      new_fields.push_back(op1_f);
    }
    else {
      new_fields.push_back(getUnionRange(std::static_ptr_cast<ValueInfoWithRange>(op1_f),
                                         std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(op2_s->getField(i))));
    }
  }
  return std::make_shared<StructInfo>(new_fields);
}

std::shared_ptr<ValueInfoWithRange> taffo::fillRangeHoles(const std::shared_ptr<ValueInfoWithRange>& src,
                                                          const std::shared_ptr<ValueInfoWithRange>& dst) {
  if (!src)
    return copyRange(dst);
  if (!dst || std::isa_ptr<ScalarInfo>(src))
    return copyRange(src);
  const std::shared_ptr<StructInfo> src_s = std::static_ptr_cast<StructInfo>(src);
  const std::shared_ptr<StructInfo> dst_s = std::static_ptr_cast<StructInfo>(dst);
  SmallVector<std::shared_ptr<ValueInfo>, 4U> new_fields;
  unsigned num_fields = src_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    if (const std::shared_ptr<PointerInfo> ptr_field = std::dynamic_ptr_cast_or_null<PointerInfo>(src_s->getField(i))) {
      new_fields.push_back(std::make_shared<PointerInfo>(ptr_field->getPointed()));
    }
    else if (i < dst_s->getNumFields()) {
      new_fields.push_back(fillRangeHoles(std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(src_s->getField(i)),
                                          std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(dst_s->getField(i))));
    }
  }
  return std::make_shared<StructInfo>(new_fields);
}
