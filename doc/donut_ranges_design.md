# Donut Ranges in VRA — Design Note

*COT project A.Y. 2025/26, Mohamed (solo).*
*Combined work plan: TAFFO sub-projects #1 "Donut Ranges in VRA" and*
*#3 "Range Models for NN Activation Functions".*

---

## 1. Motivation

TAFFO's Value Range Analysis (VRA) currently represents each SSA value's
range as a single closed interval `[min, max]`. This is fine for simple
loops but loses precision in two common situations:

1. **Bimodal distributions.** Neural network weights trained with L2
   regularisation and zero-mean init often cluster in two clouds around
   ±µ and are *never near zero*. A classic interval is forced to include
   the region around zero, which defeats the whole point of picking a
   tight fixed-point format.

2. **Division by an interval that crosses zero.** TAFFO currently sidesteps
   this with a `DIV_EPS = 1e-8` hack that nudges the divisor away from 0,
   producing enormously wide output ranges. If the *actual* divisor is
   known to exclude an interval around zero, the division would be trivial.

**Donut ranges** are a minimal extension to VRA that fixes both problems:
a donut range is a canonicalised union of disjoint closed intervals
`[lo_0, hi_0] ∪ [lo_1, hi_1] ∪ … ∪ [lo_{N-1}, hi_{N-1}]`, subject to a
small fixed upper bound `N ≤ kMaxComponents` for decidability.

A donut with `N = 1` is indistinguishable from a classic interval, so
donut ranges subsume the existing abstract domain.

## 2. Non-goals

* Full polyhedral or zonotope domain — too heavy for a course project.
* Arbitrary N — we cap components and widen when we exceed the cap.
* Any API change for downstream passes. DTA and Conversion must see the
  convex hull `[min, max]` unchanged.

## 3. Data model

We extend `struct taffo::Range` in `lib/TaffoCommon/TaffoInfo/RangeInfo.hpp`
with an **optional** list of disjoint components:

```cpp
struct Range : public Serializable, public tda::Printable {
  // existing fields — always reflect the convex hull
  double min = -INF, max = +INF;
  llvm::APFloat lower, upper;
  bool isLowerNegInf = true, isUpperPosInf = true;
  const llvm::fltSemantics* sem;

  // NEW: donut components.
  // When EMPTY -> classic interval [min, max].
  // When NON-EMPTY -> canonicalised sorted list of disjoint closed intervals
  // whose convex hull is exactly [min, max].
  std::vector<std::pair<double, double>> components;

  static constexpr unsigned kMaxComponents = 4;

  bool isDonut() const { return components.size() >= 2; }
  void addComponent(double lo, double hi);
  void canonicalize();
  void rebuildHullFromComponents();
};
```

### Invariants (after canonicalisation)

1. `components` is sorted by `.first`.
2. No two consecutive components touch or overlap (`next.first > cur.second`).
3. `components.size() <= kMaxComponents`.
4. If `components.size() == 1`, `components.clear()` — a 1-component donut is
   exactly a classic interval, so we store it as such to keep legacy paths
   fast.
5. When `!components.empty()`, `min == components.front().first` and
   `max == components.back().second`.

### Canonicalisation algorithm

```
sort components by .first
sweep: merge any (cur, next) where next.first <= cur.second
while components.size() > kMaxComponents:
    find the pair (i, i+1) with the smallest gap components[i+1].first - components[i].second
    merge them
if components.size() == 1: clear()
rebuild hull from components
```

Merging adjacent pairs (rather than always merging the first two) preserves
the “donut-ness” of the representation: we throw away the least-informative
hole first.

## 4. Arithmetic

For each of `handleAdd/Sub/Mul/Div`:

* **Both operands classic:** existing code path, unchanged. Zero overhead.
* **Either operand is a donut:** compute the Cartesian product of
  components, apply the classic per-interval operator to each pair, collect
  results into a new Range, canonicalise.

Example (division, the killer feature):

```
a = [2, 4]
b = [-0.5, -0.1] ∪ [0.1, 0.4]
a / b:
   [2, 4] / [-0.5, -0.1] = [-40, -4]
   [2, 4] / [0.1, 0.4]   = [5, 40]
result: [-40, -4] ∪ [5, 40]
```

No `DIV_EPS` fudge. The zero-excluding structure of `b` is preserved.

## 5. Join (union)

The existing convex-hull `getUnionRange` is extended:

* Classic ∪ Classic → single interval covering both (when they touch or
  overlap) OR a donut with 2 components (when they don't).
* Donut ∪ X → concatenate components (unwrapping X if it's classic) and
  canonicalise.

## 6. Activation functions

If an activation handler receives a donut-range input, apply the handler
piecewise to each component and union the results. This is the payoff that
couples the two sub-projects:

```
sigmoid([-10, -5] ∪ [5, 10])
 = sigmoid([-10, -5]) ∪ sigmoid([5, 10])
 ≈ [4.5e-5, 6.7e-3] ∪ [0.993, 0.99995]
```

Without donut tracking, we are forced to compute
`sigmoid([-10, 10]) ≈ [4.5e-5, 0.99995]`, which is useless for DTA.

## 7. Opt-in

A global flag `Range::enableDonut` (default `false`) and a corresponding
LLVM `cl::opt` `-vra-donut-ranges` gate every new path. When the flag is
off:

* `addComponent` is a no-op.
* Every arithmetic operator falls through to the classic path.
* Serialisation omits the `components` field.

This guarantees that existing TAFFO users see **no regression** unless they
explicitly ask for donut ranges.

## 8. Serialisation

JSON gains an optional `components` field:

```json
{ "min": -0.5, "max": 0.4, "components": [[-0.5, -0.1], [0.1, 0.4]] }
```

`deserialize` reads `components` only if present, preserving backward
compatibility with older metadata files.

## 9. Testing strategy

Unit tests (under `unittests/`):

* Canonicalisation: sort, merge overlap, merge adjacency, clip to
  `kMaxComponents`, collapse 1-component → classic.
* Arithmetic: add / sub / mul / div on donuts, including the Cartesian
  product correctness.
* Division by donut excluding zero: no `DIV_EPS` blow-up.
* Activation piecewise application: sigmoid and GELU over donut inputs.
* Serialisation round-trip with and without components.

## 10. Benchmark strategy (later phase)

Build a small quantised MLP (e.g. 784-128-10 on MNIST) and measure:

1. Baseline TAFFO — classic ranges only.
2. TAFFO + activation function handlers only.
3. TAFFO + donut ranges only.
4. TAFFO + donut ranges + activation function handlers.

Report, per variable: bit-widths assigned by DTA, total binary size, and
end-to-end inference accuracy vs IEEE-float reference. Expected result:
the combined (4) configuration assigns strictly fewer bits than any of
(1-3) without degrading accuracy.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Donut arithmetic fan-out blows up | Hard cap `kMaxComponents`, gap-merging widening |
| Downstream passes crash on new metadata | Convex hull always in sync; `components` is optional; opt-in flag |
| GELU/SiLU minimum constants become unsound under future refactor | Hard-coded over-approximations; unit tests pin them |
| Supervisors prefer a different sub-project split | Work is a strict superset of sub-projects #1 and #3; either can be presented in isolation |

## 12. What is *not* in this design note

* The concrete LaTeX report (phase 6).
* Per-variable bit-width results (phase 5).
* MLIR-side TAFFO integration (explicitly out of scope).
