#include "RangeInfo.hpp"

#include <algorithm>
#include <limits>

using namespace llvm;
using namespace taffo;

// Global opt-in for donut ranges. Flipped by the -vra-donut-ranges cl::opt
// in ValueRangeAnalysisPass. Default OFF means existing TAFFO users see
// zero change.
bool Range::enableDonut = false;

Range::Range(const llvm::fltSemantics& S, llvm::APFloat l, llvm::APFloat u) : sem(&S), lower(llvm::APFloat::getZero(S, false)), upper(llvm::APFloat::getZero(S, false)) {

  if (&l.getSemantics() != sem) {
    bool lose = false;
    l.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &lose);
  }
  if (&u.getSemantics() != sem) {
    bool lose = false;
    u.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &lose);
  }
  lower = std::move(l);
  upper = std::move(u);
  isLowerNegInf = false;
  isUpperPosInf = false;

  normalizeOrder();
  toDouble();
}

Range::Range(double min, double max) : min(min), max(max), sem(&llvm::APFloat::IEEEdouble()), lower(llvm::APFloat::getZero(*sem, false)), upper(llvm::APFloat::getZero(*sem, false)) {
  if (min > max) std::swap(min, max);
  toAPFloat();
  toDouble();
}

void Range::setLower(const APFloat& l) {
  lower = l;
  if (&lower.getSemantics() != sem) {
    bool lose = false;
    lower.convert(*sem, APFloat::rmNearestTiesToEven, &lose);
  }
  isLowerNegInf = false;
  normalizeOrder();
  toDouble();
}
void Range::setUpper(const APFloat& u) {
  upper = u;
  if (&upper.getSemantics() != sem) {
    bool lose = false;
    upper.convert(*sem, APFloat::rmNearestTiesToEven, &lose);
  }
  isUpperPosInf = false;
  normalizeOrder();
  toDouble();
}

void Range::toDouble() {
  min = isLowerNegInf ? -std::numeric_limits<double>::infinity() : lower.convertToDouble();
  max = isUpperPosInf ? +std::numeric_limits<double>::infinity() : upper.convertToDouble();
  if (min > max) std::swap(min, max);
}

void Range::toAPFloat() {
  if (std::isinf(min) && min < 0) isLowerNegInf = true;
  else {
    isLowerNegInf = false;
    APFloat l(min);
    bool lose = false;
    l.convert(*sem, APFloat::rmTowardNegative, &lose);
    lower = std::move(l);
  }

  if (std::isinf(max) && max > 0) isUpperPosInf = true;
  else {
    isUpperPosInf = false;
    APFloat u(max);
    bool lose = false;
    u.convert(*sem, APFloat::rmTowardPositive, &lose);
    upper = std::move(u);
  }
  normalizeOrder();
}

bool Range::isExponentialMonotonic() {
  return min > 1.0 || max < -1.0;
}

bool Range::isOneModular() {
  return (min > 0.0 && max < 1.0) || (max < 0.0 && min > -1.0);
}

bool Range::cross(const APFloat& v) const {
  if (v.isNaN()) return false;
  APFloat x = v;
  if (&x.getSemantics() != sem) {
    bool lose = false;
    x.convert(*sem, APFloat::rmNearestTiesToEven, &lose);
  }
  
  return (isLowerNegInf || x.compare(lower) != APFloat::cmpLessThan) && 
          (isUpperPosInf || x.compare(upper) != APFloat::cmpGreaterThan);
}

bool Range::cross(double val) const { 
  return min <= val && max >= val;
}

std::string Range::toString() const {
  std::stringstream ss;
  if (!components.empty()) {
    // Donut form: print every disjoint component joined by 'U'.
    for (size_t i = 0; i < components.size(); ++i) {
      if (i > 0) ss << " U ";
      ss << "[" << components[i].first << ", " << components[i].second << "]";
    }
    return ss.str();
  }
  if (isLowerNegInf) ss << "[-inf, ";
  else ss << "[" << min << ", ";
  if (isUpperPosInf) ss << "inf]";
  else ss << max << "]";
  return ss.str();
}

json Range::serialize() const {
  json j;
  j["min"] = serializeDouble(isLowerNegInf ? -std::numeric_limits<double>::infinity() : min);
  j["max"] = serializeDouble(isUpperPosInf ? std::numeric_limits<double>::infinity() : max);
  // The `components` field is written only when the range is actually a
  // donut. This keeps metadata backwards-compatible with older TAFFO
  // deserialisers which do not know about donut ranges.
  if (!components.empty()) {
    json arr = json::array();
    for (const auto& c : components) {
      arr.push_back({serializeDouble(c.first), serializeDouble(c.second)});
    }
    j["components"] = arr;
  }
  return j;
}

void Range::deserialize(const json& j) {
  min = deserializeDouble(j["min"]);
  max = deserializeDouble(j["max"]);
  components.clear();
  if (j.contains("components") && j["components"].is_array()) {
    for (const auto& entry : j["components"]) {
      if (entry.is_array() && entry.size() == 2) {
        components.emplace_back(deserializeDouble(entry[0]),
                                deserializeDouble(entry[1]));
      }
    }
    canonicalize();
  }
  normalizeOrder();
  toAPFloat();
}

void Range::normalizeOrder() {
  if (!isLowerNegInf && !isUpperPosInf && lower.compare(upper) == APFloat::cmpGreaterThan) {
    std::swap(lower, upper);
  }
}

// ===== DONUT RANGES =====

void Range::addComponent(double lo, double hi) {
  // If donut tracking is disabled, preserve legacy behaviour: the caller
  // should set min/max directly. We silently widen the existing hull so
  // that even if a stray addComponent slips through with the flag off we
  // do not lose information.
  if (lo > hi) std::swap(lo, hi);
  if (!enableDonut) {
    if (std::isnan(min) || lo < min) min = lo;
    if (std::isnan(max) || hi > max) max = hi;
    return;
  }
  components.emplace_back(lo, hi);
  canonicalize();
}

void Range::canonicalize() {
  if (components.empty()) return;

  // 1. Drop any NaN / degenerate entries and fix swapped endpoints.
  components.erase(
      std::remove_if(components.begin(), components.end(),
                     [](const std::pair<double, double>& c) {
                       return std::isnan(c.first) || std::isnan(c.second);
                     }),
      components.end());
  for (auto& c : components) {
    if (c.first > c.second) std::swap(c.first, c.second);
  }

  if (components.empty()) return;

  // 2. Sort by lower bound.
  std::sort(components.begin(), components.end(),
            [](const std::pair<double, double>& a,
               const std::pair<double, double>& b) {
              return a.first < b.first;
            });

  // 3. Single merge sweep over touching / overlapping neighbours.
  std::vector<std::pair<double, double>> merged;
  merged.reserve(components.size());
  merged.push_back(components.front());
  for (size_t i = 1; i < components.size(); ++i) {
    auto& last = merged.back();
    const auto& cur = components[i];
    if (cur.first <= last.second) {
      // Touching or overlapping -> merge into last.
      if (cur.second > last.second) last.second = cur.second;
    } else {
      merged.push_back(cur);
    }
  }
  components = std::move(merged);

  // 4. If we are still above the budget, widen by repeatedly merging the
  // pair of neighbours with the smallest gap between them. This throws
  // away the least informative hole first.
  while (components.size() > kMaxComponents) {
    size_t bestIdx = 0;
    double bestGap = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < components.size(); ++i) {
      const double gap = components[i + 1].first - components[i].second;
      if (gap < bestGap) {
        bestGap = gap;
        bestIdx = i;
      }
    }
    components[bestIdx].second =
        std::max(components[bestIdx].second, components[bestIdx + 1].second);
    components.erase(components.begin() + bestIdx + 1);
  }

  // 5. A 1-component donut is just a classic interval; drop the vector so
  // downstream fast paths kick in.
  if (components.size() == 1) {
    const auto c = components.front();
    components.clear();
    min = c.first;
    max = c.second;
    toAPFloat();
    return;
  }

  // 6. Sync convex hull to the first and last surviving components.
  rebuildHullFromComponents();
}

void Range::rebuildHullFromComponents() {
  if (components.empty()) return;
  min = components.front().first;
  max = components.back().second;
  if (std::isinf(min) && min < 0) isLowerNegInf = true;
  else isLowerNegInf = false;
  if (std::isinf(max) && max > 0) isUpperPosInf = true;
  else isUpperPosInf = false;
  toAPFloat();
}

std::vector<std::pair<double, double>>
Range::getComponentsOrHull() const {
  if (!components.empty()) return components;
  return {{min, max}};
}

Range Range::meet(const Range& R) const {
  Range out(*sem);

  if (isLowerNegInf && R.isLowerNegInf) out.isLowerNegInf = true;
  else {
    out.isLowerNegInf = false;
    if (isLowerNegInf) {
      out.lower = R.lower;
    } else if (R.isLowerNegInf) {
      out.lower = lower;
    } else {
      out.lower = lower.compare(R.lower) == APFloat::cmpGreaterThan ? lower : R.lower;
    }
  }

  if (isUpperPosInf && R.isUpperPosInf) out.isUpperPosInf = true;
  else {
    out.isUpperPosInf = false;
    if (isUpperPosInf) {
      out.upper = R.upper;
    } else if (R.isUpperPosInf) {
      out.upper = upper;
    } else {
      out.upper = upper.compare(R.upper) == APFloat::cmpLessThan ? upper : R.upper;
    }
  }

  out.normalizeOrder();
  out.toDouble();
  return out;
}

Range Range::join(const Range& R) const {
  Range out(*sem);

  if (isLowerNegInf || R.isLowerNegInf) out.isLowerNegInf = true;
  else {
    out.isLowerNegInf = false;
    out.lower = lower.compare(R.lower) == APFloat::cmpLessThan ? lower : R.lower;
  }

  if (isUpperPosInf || R.isUpperPosInf) out.isUpperPosInf = true;
  else {
    out.isUpperPosInf = false;
    out.upper = upper.compare(R.upper) == APFloat::cmpGreaterThan ? upper : R.upper;
  }

  out.normalizeOrder();
  out.toDouble();

  // Donut-aware join: if either operand carries disjoint components, union
  // them and canonicalise. When enableDonut is off, canonicalize() still
  // merges them but the caller won't see new components because the
  // legacy path above already filled in the convex hull.
  if (enableDonut && (!components.empty() || !R.components.empty())) {
    const auto lhs = getComponentsOrHull();
    const auto rhs = R.getComponentsOrHull();
    out.components.reserve(lhs.size() + rhs.size());
    for (const auto& c : lhs) out.components.push_back(c);
    for (const auto& c : rhs) out.components.push_back(c);
    out.canonicalize();
  }

  return out;
}

Range Range::widen(const Range& R) const {
  Range out(*sem);

  if (!isLowerNegInf && !R.isLowerNegInf && R.lower.compare(lower) == APFloat::cmpLessThan) {
    out.isLowerNegInf = true;
  } else {
    out.isLowerNegInf = isLowerNegInf;
    out.lower = lower;
  }

  if (!isUpperPosInf && !R.isUpperPosInf && R.upper.compare(upper) == APFloat::cmpGreaterThan) {
    out.isUpperPosInf = true;
  } else {
    out.isUpperPosInf = isUpperPosInf;
    out.upper = upper;
  }

  out.normalizeOrder();
  out.toDouble();
  return out;
}

Range Range::narrow(const Range& R) const {
  Range out(*sem);

  if (isLowerNegInf && !R.isLowerNegInf) {
    out.lower = R.lower;
  } else {
    out.lower = lower;
    out.isLowerNegInf = isLowerNegInf;
  }

  if (isUpperPosInf && !R.isUpperPosInf) {
    out.upper = R.upper;
  } else {
    out.upper = upper;
    out.isUpperPosInf = isUpperPosInf;
  }

  out.normalizeOrder();
  out.toDouble();
  return out;
}

std::shared_ptr<Range> Range::meet(const std::shared_ptr<Range>& R) const {
  if (!R) return clone();
  auto res = meet(*R);
  return std::make_shared<Range>(std::move(res));
}

std::shared_ptr<Range> Range::join(const std::shared_ptr<Range>& R) const {
  if (!R) return clone();
  auto res = join(*R);
  return std::make_shared<Range>(std::move(res));
}

std::shared_ptr<Range> Range::widen(const std::shared_ptr<Range>& R) const {
  if (!R) return clone();
  auto res = widen(*R);
  return std::make_shared<Range>(std::move(res));
}

std::shared_ptr<Range> Range::narrow(const std::shared_ptr<Range>& R) const {
  if (!R) return clone();
  auto res = narrow(*R);
  return std::make_shared<Range>(std::move(res));
}

Range Range::Point(const llvm::APFloat& V) {
  return Range(V,V);
}

Range Range::Top() {
  return Range();
}
