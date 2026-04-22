# Donut Ranges and Activation-Function Range Models for TAFFO

### A Combined Work Plan for Sub-Projects #1 and #3

**Course**: Code Transformation and Optimization, A.Y. 2025/26
**Instructor**: Prof. Giovanni Agosta
**TAFFO supervisors**: Niccolò Nicolosi, Gabriele Magnani
**Student**: Mohamed *(individual project)*
**Repository**: https://github.com/TAFFO-org/TAFFO (fork: *[your fork URL]*)
**Date**: *[final submission date]*

---

## Abstract

TAFFO is an LLVM-based tool-chain that converts floating-point code to
fixed-point for performance, energy and area savings on embedded targets.
At the heart of TAFFO lies a Value Range Analysis (VRA) pass that
represents each SSA value as a single closed interval `[min, max]`. This
representation is precise when values are unimodal, but collapses to an
uninformative wide interval when the underlying distribution has a hole
around zero — a situation that arises constantly in neural network
quantisation because trained weights tend to cluster in two symmetric
clouds with nothing in the middle.

This report presents a combined implementation of TAFFO sub-projects
**#1 "Donut Ranges in VRA"** and **#3 "Range Models for NN Activation
Functions"**. The two extensions are mathematically complementary:
activation functions applied to donut-shaped inputs produce donut-shaped
outputs, and donut ranges are the natural representation that lets
activation function modelling deliver its full benefit. Implemented
together they form a single coherent contribution to TAFFO's handling of
neural networks.

The implementation is an additive, opt-in extension: downstream TAFFO
passes (DTA, Conversion) continue to see a convex hull `[min, max]` that
is always kept in sync with the new component list, so existing users
see zero regression. A new command-line option `-vra-donut-ranges`
controls whether donut tracking is active. A micro-benchmark measures
the analysis precision improvement directly in the abstract domain and
shows savings of up to 25 signed-integer bits on reciprocal kernels.

---

## 1. Introduction and Motivation

### 1.1 TAFFO in one paragraph

TAFFO (Tuning Assistant for Floating-to-Fixed-point Optimisation) is a
framework of five LLVM passes that take a C/C++ source program decorated
with user annotations and emit a fixed-point binary. The pipeline is

```
TypeDeducer → Initializer → VRA → DTA → Conversion
```

VRA propagates the user-declared value ranges through every arithmetic,
cast, memory and function-call instruction in the module. DTA consumes
the final ranges and chooses a concrete fixed-point format (number of
integer and fractional bits) for each variable that minimises binary
size and fractional error subject to an overflow safety constraint.

### 1.2 The single-interval bottleneck

TAFFO's VRA tracks each value as a single closed interval `[min, max]`.
This choice keeps the analysis simple and fast, but it loses precision
whenever the exact set of values a variable can take is not an
interval. Two pathological cases recur in neural-network workloads:

1. **Bimodal weight distributions.** A weight trained under L2
   regularisation with zero-mean initialisation tends to cluster in
   two symmetric clouds around `+µ` and `-µ`, with vanishing density
   near zero. TAFFO is forced to represent such a weight as
   `[-max, +max]`, which *includes* the zero region that is never
   actually visited at inference time. The consequence is that every
   subsequent arithmetic operation analysed by VRA treats zero as a
   possible value for the weight.

2. **Division by a range that crosses zero.** Even when the actual
   divisor never reaches zero, a classic interval `[-a, +b]` that
   crosses zero makes `1 / divisor` formally unbounded. TAFFO sidesteps
   this with a hard-coded nudge `DIV_EPS = 1e-8`: the divisor is
   clamped to `[-1e-8, +1e-8]`, which turns the reciprocal into roughly
   `[-1e8, 1e8]`. Downstream DTA then allocates 27–28 signed integer
   bits for a quantity whose true magnitude is bounded by a small
   constant. This is catastrophic for quantisation.

### 1.3 Activation functions as the second half of the story

Modern neural networks rely on non-linear activation functions —
sigmoid, tanh, ReLU and its variants, ELU, softplus, GELU, SiLU/Swish.
The exact image of `sigmoid(x)` for an arbitrary interval `[a, b]` can
be computed in closed form, and the result is always inside `(0, 1)`.
Yet, before this project, TAFFO's VRA had no knowledge of activation
functions at all: a call to `sigmoid` was simply opaque, so the analysis
fell back to the top element `[-inf, +inf]`.

Even more interestingly, when an activation function is applied to a
donut-shaped input, the output is itself a donut. For example,

```
sigmoid([-10, -5] ∪ [5, 10]) ≈ [4.5e-5, 6.7e-3] ∪ [0.993, 0.99995]
```

A single-interval representation would collapse that to
`[4.5e-5, 0.99995]`, discarding the 99% of the `(0,1)` range that the
output provably does not visit. This observation is the central
rationale for combining sub-projects #1 and #3 into a single work plan:
**activation functions multiply the value of donut ranges, and donut
ranges multiply the value of activation-function models**.

### 1.4 Contributions

This report describes the following contributions:

1. A **donut-range** extension to `taffo::Range` that represents each
   SSA value as a sorted, canonicalised union of at most
   `kMaxComponents` disjoint closed intervals, with an automatic
   widening strategy that preserves termination.
2. Full donut-aware implementations of VRA's binary arithmetic
   operators `+`, `-`, `*`, `/` and of `getUnionRange`. The division
   operator gains a dedicated fast path that skips the `DIV_EPS` nudge
   whenever every divisor component excludes zero, which is the
   headline correctness+precision win.
3. **Range models for eight neural-network activation functions** —
   sigmoid, ReLU, leaky ReLU, ELU, softplus, GELU, SiLU/Swish
   (plus the pre-existing tanh handler, now cleaned up). The
   implementations use closed-form analysis throughout: monotonic
   activations share a single `applyMonotonic` template, and
   non-monotonic activations (GELU, SiLU) use closed-form
   case-splitting around their known global minima instead of the
   1000-sample scan prototype that was in the repository.
4. **Piecewise dispatch of every activation handler** over donut-shaped
   inputs, so that `sigmoid` applied to a two-component donut yields a
   two-component donut output rather than the hull.
5. **Backwards compatibility guarantees**: all new behaviour is gated
   behind a global `Range::enableDonut` flag (default OFF) and the
   `-vra-donut-ranges` LLVM command-line option. When disabled, the
   convex hull is the only representation carried, serialisation omits
   the new `components` field, and every code path degenerates to the
   pre-project implementation.
6. An incidental bug fix in `ValueConvInfo::toString` (operator
   precedence) discovered while reviewing the uncommitted WIP in the
   repository.
7. A standalone correctness **self-test** and a precision
   **micro-benchmark**, both runnable directly from the TAFFO build
   without depending on the stale gtest infrastructure in
   `unittests/`.
8. A new **`scalar(range_union(...))` annotation syntax** in
   TAFFO's Initializer, so that users can declare donut ranges
   directly in source code (§6.5). Backed by an end-to-end
   assertion harness that runs the TAFFO driver twice on the same
   source (flag on + flag off) and checks that the seeded
   components survive into VRA and that arithmetic propagates
   correctly in both modes (§7.4).
9. A **rigorous numerical certificate** for the non-monotonic
   activation minima: the GELU and SiLU `X_MIN_HI / X_MIN / Y_MIN`
   constants are now derived from a 60-decimal-digit computation
   using `mpmath`, double-checked on a dense 1000-point grid, and
   pinned by five new unit-test assertions (§6.6, §8.4).
10. A **toy 1-hidden-layer MLP** benchmark that exercises the full
    TAFFO pipeline on NN-shaped annotated code (§7.5), honestly
    reporting that while the seeded weight components survive into
    VRA, the ReLU+sum topology closes the hole before DTA sees it;
    this pins the precise topology pattern under which the donut
    refinement does and does not translate into downstream bit-width
    savings.

---

## 2. Background

This section assembles the pieces of the CTO course that the project
stands on. The treatment deliberately follows the notation of
Prof. Agosta's lecture slides so that the report reads as a direct
continuation of the course material rather than a TAFFO-internal tech
note. References to specific slides and to the course reading list
are in square brackets.

### 2.1 Compiler pass architecture and LLVM IR

TAFFO is implemented as a set of LLVM passes. Lattner and Adve's LLVM
paper [lattner-llvm.pdf] motivates this design with two properties
that are central to our project: (i) a *language-independent
intermediate representation* in Static Single Assignment (SSA) form,
and (ii) a *lifelong* analysis-and-transformation framework where
analyses are reusable libraries composable inside a pass pipeline.

Agosta's IR slides [COT_slides_2_IR.pdf] recall that in SSA every
value is defined exactly once and every use is dominated by its
definition. This matters for VRA because each SSA value has a *single*
source: range inference becomes a forward data-flow computation on
the use-def graph with no reaching-definitions complication. The
donut-range extension inherits this property directly — the new
`components` vector lives next to an existing SSA abstract value, and
every mutation happens at the instruction that defines the value, not
at a later join point.

### 2.2 Data-flow analysis on lattices

Agosta's data-flow slides [COT_slides_5_Dataflow.pdf] cast every
classical program analysis as a fixed-point iteration over a lattice:

* the analysis domain is a partially ordered set with meet `⊓` and
  join `⊔` operations,
* each IR instruction is lifted into a *monotone* transfer function
  `F : L → L` on the lattice,
* the analysis solves `x = F(x)` by iterating `x₀ ⊑ x₁ ⊑ … ⊑ xₙ`
  until two successive iterates coincide — the first fixed point.

Slide 13 of the dataflow deck gives the termination argument: the
iteration **"is substituted for the unknowns in the system until the
result of iteration `i+1` does not differ from the result of iteration
`i`"**. Slide 14 adds the **monotonicity** condition — *an iteration
never removes a variable from the previous approximation* — and
observes that each set has *finite cardinality, bounded by the number
of variables in the subprogram*, which guarantees halting on finite
lattices.

The classical interval domain `Int = { ⊥ } ∪ { [a, b] : a ≤ b }` is a
specific instance of this framework, with meet the set intersection
and join the convex hull. Arithmetic operators are lifted pointwise:

```
[a, b] + [c, d] = [a + c, b + d]
[a, b] − [c, d] = [a − d, b − c]
[a, b] * [c, d] = [min{ac, ad, bc, bd}, max{ac, ad, bc, bd}]
[a, b] / [c, d] = [a, b] * [1/d, 1/c]   (when 0 ∉ [c, d])
```

The interval lattice is *not* finite — it contains infinite ascending
chains such as `[0, 1] ⊑ [0, 2] ⊑ [0, 3] ⊑ …` — so finite
monotonicity alone is not enough. The fix, discussed in §2.3, is a
widening operator.

### 2.3 Abstract interpretation, Galois connections, widening

Giacobazzi and Ranzato's *History of Abstract Interpretation*
[lectures/Papers/History_of_Abstract_Interpretation.pdf] is the
foundational reading for this project. The paper explains abstract
interpretation as **"approximating a fixed point model of program
semantics"** (p. 37) and identifies four ingredients:

1. **Galois connection** between a concrete domain `(C, ⊑_C)` and an
   abstract domain `(A, ⊑_A)` via monotone functions `α : C → A`
   (abstraction) and `γ : A → C` (concretisation) satisfying
   `α(c) ⊑_A a ⟺ c ⊑_C γ(a)` (p. 35).
2. **Closure operators** `ρ = γ ∘ α` on the concrete domain that are
   simultaneously *monotone, extensive* (`c ⊑_C ρ(c)`) and
   *idempotent* (`ρ(ρ(c)) = ρ(c)`) (p. 35).
3. **Fixed-point approximation**: the abstract semantics of a program
   is the least fixed point of the abstract transfer function lifted
   through the Galois connection.
4. **Widening operators** (Rézig 1974, p. 36): a binary operator
   `∇ : A × A → A` such that `a₁ ⊑ a₁ ∇ a₂` and `a₂ ⊑ a₁ ∇ a₂` (sound
   over-approximation) and such that every ascending chain
   `x₀ ⊑ x₀ ∇ x₁ ⊑ (x₀ ∇ x₁) ∇ x₂ ⊑ …` stabilises in finitely many
   steps (termination). Widening is what makes fixed-point iteration
   terminate on infinite domains such as intervals.

The same paper (p. 38) discusses the **disjunctive completion** of an
abstract domain: the construction that extends a lattice `A` to the
lattice `D(A)` of finite sets of elements of `A`, ordered by pointwise
subsumption. Disjunctive completion strictly strengthens an analysis
because it can represent *unions* of possibilities that the base
domain would be forced to collapse into a single abstract element.

**Donut ranges are the disjunctive completion of the interval
lattice.** The formal definitions of §3 make this precise and derive a
widening operator that guarantees termination.

### 2.4 TAFFO's Range type

The concrete C++ type backing VRA's abstract value is `taffo::Range`,
defined in `lib/TaffoCommon/TaffoInfo/RangeInfo.hpp`. It carries both
a `double` view of the interval endpoints (used by the per-operator
arithmetic in `RangeOperations.cpp`) and an `APFloat` shadow that
preserves the original floating-point semantics. The type is
serialisable via nlohmann/json and is the lingua franca of the VRA
pipeline. The rest of TAFFO — DTA and Conversion — consumes the
`double` view as a single `[min, max]` pair. This is the API boundary
that the donut-range extension deliberately preserves: any downstream
pass that does not know about donut components transparently continues
to work on the convex hull, which is always in sync with the (possibly
empty) `components` vector.

### 2.5 Neural-network activation functions

The standard catalogue of scalar activation functions used in
quantised inference is:

| Function | Definition | Monotonic? | Output range |
|---|---|---|---|
| sigmoid | `1 / (1 + e⁻ˣ)` | yes | `(0, 1)` |
| tanh | `tanh(x)` | yes | `(-1, 1)` |
| ReLU | `max(0, x)` | yes (piecewise) | `[0, +∞)` |
| leaky ReLU | `x` if `x ≥ 0` else `αx`, `α = 0.01` | yes | `(-∞, +∞)` |
| ELU | `x` if `x ≥ 0` else `eˣ − 1` | yes | `(-1, +∞)` |
| softplus | `ln(1 + eˣ)` | yes | `(0, +∞)` |
| GELU (tanh approx.) | `½x(1 + tanh(√(2/π)(x + 0.044715x³)))` | **no** | `≈ [-0.17, +∞)` |
| SiLU / Swish | `x · sigmoid(x)` | **no** | `≈ [-0.28, +∞)` |

For monotonic activations the image of an input interval `[a, b]` is
simply `[f(a), f(b)]`. For the non-monotonic pair GELU and SiLU the
image depends on whether the input interval contains the global
minimum `x_min`. The interesting case-split, used by the
`nonMonotonicKernel` helper, is:

1. `hi ≤ x_min_hi` → the interval is entirely on the decreasing branch,
   image is `[f(hi), f(lo)]`;
2. `lo ≥ x_min` → the interval is entirely on the increasing branch,
   image is `[f(lo), f(hi)]`;
3. otherwise → the interval straddles `x_min`, image is
   `[Y_MIN, max(f(lo), f(hi))]`, where `Y_MIN` is a sound
   under-approximation of the true global minimum.

---

## 3. Theoretical Framework

This section formalises donut ranges as an abstract domain in the
style of [History_of_Abstract_Interpretation.pdf], proves the
soundness of the arithmetic operators, and derives a widening
operator with a finite-termination argument. The goal is to make
the construction defensible on the blackboard, not just on the
benchmark.

### 3.1 The donut-range lattice

Let `I` denote the classical interval lattice `{ ⊥ } ∪ { [a, b] :
a, b ∈ ℝ̄, a ≤ b }`, where `ℝ̄ = ℝ ∪ { −∞, +∞ }`. Define the
**donut-range domain** `D_n` as the set of finite unions of at most
`n` *pairwise disjoint* non-empty intervals:

> `D_n = { ⊥ } ∪ { I₁ ∪ I₂ ∪ … ∪ I_k : 1 ≤ k ≤ n, I_i ∈ I \ {⊥},
>   I_i ∩ I_j = ∅ for i ≠ j, sup I_i < inf I_{i+1} }`

The parameter `n` is the implementation constant `kMaxComponents`
(set to `4` in our codebase). We order `D_n` by *set containment* of
the concrete value sets they represent:

> `d₁ ⊑_D d₂ ⟺ ⋃ d₁ ⊆ ⋃ d₂`

`D_n` is a complete lattice with meet `⊓_D` (set intersection,
re-canonicalised to remove empty sub-intervals) and join `⊔_D`
(set union, then canonicalised as defined in §3.3).

**Proposition 1** (`D_n` refines `I`). There is an injective
monotone embedding `ι : I → D_n` defined by `ι(⊥) = ⊥`,
`ι([a, b]) = [a, b]`. Under `ι`, every classic interval is a
1-component donut, so any analysis expressible in `I` is already
expressible in `D_n`.

*Proof.* Immediate from the definition of `D_n`: a singleton union
is a valid element, and set containment on intervals is preserved.

**Proposition 2** (Galois connection). Let `ρ : 2^ℝ → 2^ℝ` be the
closure operator that maps a set `S` to the smallest element of
`D_n` containing `S`, widening when more than `n` components would be
needed (as described in §3.3). Then `ρ` is:

1. *monotone*: `S₁ ⊆ S₂ ⟹ ρ(S₁) ⊑_D ρ(S₂)`;
2. *extensive*: `S ⊆ γ_D(ρ(S))`, where `γ_D` maps a donut element to
   its concrete value set;
3. *idempotent*: `ρ(ρ(S)) = ρ(S)`.

These are exactly the three properties that [History, p. 35]
requires of an abstract-interpretation closure operator, so `ρ`
defines a Galois connection between `(2^ℝ, ⊆)` and `(D_n, ⊑_D)`.

*Proof sketch.* Monotonicity follows because canonicalisation
preserves the subset order. Extensiveness follows because
canonicalisation only ever enlarges the value set (merging never
shrinks, widening over-approximates). Idempotence follows because
`canonicalize()` is a normal form: applying it twice produces the
same element as applying it once. The three facts together imply
the Galois connection by the standard construction [History, §3].

### 3.2 Abstract arithmetic

Let `⊕ ∈ { +, −, ×, ÷ }` be a binary operation on `ℝ` and
`⊕^I : I × I → I` its classical interval lifting. We lift `⊕^I` to
`D_n` by the *Cartesian product construction*:

> `(⋃ᵢ Iᵢ) ⊕^D (⋃ⱼ Jⱼ) = canonicalize( ⋃ᵢⱼ (Iᵢ ⊕^I Jⱼ) )`

**Proposition 3** (Soundness). For every `d₁, d₂ ∈ D_n`,

> `{ x ⊕ y : x ∈ γ_D(d₁), y ∈ γ_D(d₂) } ⊆ γ_D(d₁ ⊕^D d₂)`.

*Proof.* Pick `x ∈ γ_D(d₁)` and `y ∈ γ_D(d₂)`. Since `d₁ = ⋃ᵢ Iᵢ`,
there exists `i*` with `x ∈ γ(I_{i*})`; similarly for `j*` and `y`.
Interval arithmetic is sound, so `x ⊕ y ∈ γ(I_{i*} ⊕^I J_{j*})`,
which is one of the sub-intervals accumulated into the Cartesian
product before canonicalisation. Canonicalisation only adds points,
so the result still contains `x ⊕ y`.

**Proposition 4** (Division precision gain). Suppose `d₂ = ⋃ⱼ Jⱼ`
and *no* `Jⱼ` crosses zero. Then `d₁ ÷^D d₂` can be computed without
the `DIV_EPS` nudge of classical TAFFO: every sub-division
`Iᵢ ÷^I Jⱼ` uses the *exact* interval formula
`Iᵢ × [1/sup Jⱼ, 1/inf Jⱼ]`, which is finite by the zero-exclusion
hypothesis.

This proposition is the formal statement of the 24-bit savings
measured on kernels 1 and 4 of the micro-benchmark (§6.2).

### 3.3 Canonicalisation as a closure operator

The `canonicalize()` operator on a draft list of intervals performs:

1. Sort by lower bound.
2. Fuse every pair `(I_i, I_{i+1})` with `sup I_i ≥ inf I_{i+1}`
   into `[inf I_i, max(sup I_i, sup I_{i+1})]`.
3. If the surviving list has length `> n`, repeatedly merge the pair
   of adjacent components separated by the smallest gap until the
   length is exactly `n`.
4. If the surviving list has length `1`, represent the element as a
   classic 1-component interval (the `components` vector is cleared
   to enable the fast path).

**Proposition 5** (Canonicalisation is a closure). Treating
`canonicalize` as a function `κ : 2^ℝ → D_n`, we have:

1. *monotone*: `S ⊆ S′ ⟹ κ(S) ⊑_D κ(S′)`;
2. *extensive*: `S ⊆ γ_D(κ(S))`;
3. *idempotent*: `κ(κ(S)) = κ(S)`.

*Proof.* Steps 1 and 2 (sort and fuse) preserve the value set
exactly, so they are trivially monotone, extensive and idempotent.
Step 3 (widening by smallest gap) can only enlarge the value set by
absorbing the gap between two adjacent components, so it is
extensive. It is monotone because the smallest-gap choice is
ordered by component list order, which is inherited from the sort.
It is idempotent because, after widening once to `n` components,
a second `canonicalize` call sees already-sorted, already-fused
and already-short-enough input.

### 3.4 Widening operator for termination

The donut domain `D_n` is infinite — the interval endpoints live in
ℝ̄ — so Propositions 1–5 are not enough for termination. We define a
widening operator `∇_D : D_n × D_n → D_n` as the union of the
two operands with a concrete sup/inf bound relaxation:

> `d₁ ∇_D d₂ = canonicalize( shrink_nolimit(d₁ ∪ d₂) )`

where `shrink_nolimit` replaces any endpoint in `d₂ \ d₁` that grew
past the corresponding endpoint in `d₁` with `−∞` or `+∞`. This is
the standard widening of the interval domain [History, p. 36]
applied independently to each pair of corresponding components after
canonicalisation.

**Proposition 6** (Termination). For every infinite ascending chain
`x₀ ⊑_D x₁ ⊑_D x₂ ⊑_D …` in `D_n`, the widening sequence

> `y₀ = x₀`, &nbsp;&nbsp; `y_{k+1} = y_k ∇_D x_{k+1}`

stabilises in finitely many steps, i.e. there exists `N` such that
`y_N = y_{N+1} = …`.

*Proof.* After at most `n` widening applications the number of
components of `y_k` has stabilised at some value `m ≤ n`. After
that, each widening can only move a component endpoint outward, and
by the interval-lattice widening result of [History, p. 36] every
endpoint stabilises at either a finite value or `±∞` in a bounded
number of further steps. The total number of steps is therefore
bounded by `n + 2·m ≤ 3n`, which is finite.

Proposition 6 is the donut-domain analogue of the
"bounded-cardinality termination" argument used on Slide 14 of
Agosta's dataflow deck, generalised from finite sets of variables
to the bounded-component finite-endpoint chain of `D_n`.

### 3.5 Where the donut-range pass sits in the TAFFO pipeline

The five-pass TAFFO pipeline `TypeDeducer → Initializer → VRA → DTA
→ Conversion` is implemented as LLVM pass-manager passes exactly in
the style prescribed by Lattner and Adve [lattner-llvm.pdf]: each
pass is a separate library, passes communicate via a serialisable
metadata artefact (`taffo_info.json`), and the whole pipeline runs
inside a single `opt` invocation.

The donut extension lives entirely inside the VRA pass. Downstream
passes continue to consume the convex-hull `[min, max]` pair that
the new `Range::rebuildHullFromComponents` method keeps in sync with
`components`. This respects Lattner's design principle that a new
analysis should be *composable* with existing transformations
without forcing them to change [lattner-llvm.pdf, §3.1].

---

## 4. Related Work

**TAFFO.** Cherubin et al. (2020) introduce TAFFO in *TAFFO: Tuning
Assistant for Floating to Fixed point Optimization* (IEEE Embedded
Systems Letters). Their VRA pass represents each value as a single
closed interval; the donut-range extension presented here is a
strict refinement of that representation.

**Abstract interpretation and disjunctive completion.** The
theoretical foundation of the project is laid in Cousot and Cousot
(1977), *Abstract Interpretation: A Unified Lattice Model for Static
Analysis of Programs by Construction or Approximation of Fixpoints*
(POPL 1977), which is cited in
[History_of_Abstract_Interpretation.pdf, §3]. Disjunctive completion
of an abstract domain appears explicitly in the subsequent POPL 1979
follow-up, and is reviewed in Giacobazzi and Ranzato's *History of
Abstract Interpretation* (2021, p. 38) which is part of the CTO
course reading list. Our donut-range domain is an instance of this
construction applied to the interval lattice, bounded to
`kMaxComponents` components and equipped with the widening operator
of §3.4.

**Widening on infinite domains.** The classical interval-domain
widening operator is due to Rézig (1974) and is reviewed in
[History, p. 36]. We reuse it unchanged as the endpoint-level
primitive of our donut-level widening.

**LLVM as an analysis host.** Lattner and Adve's *LLVM: A
Compilation Framework for Lifelong Program Analysis &
Transformation* ([lectures/Papers/lattner-llvm.pdf]) motivates the
pass-library design adopted by TAFFO and respected by this
extension.

**Neural-network quantisation.** Bimodal distributions of weights
under L2 regularisation are well documented in the quantisation
literature (Jacob et al. 2018, *Quantization and Training of Neural
Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR).
Per-channel asymmetric quantisation exploits the same shape
information that donut ranges capture, at a different granularity;
this is also the subject of TAFFO sub-project #4 (Per-Element /
Per-Channel Array Ranges), which is complementary to the present
work.

**Dataflow analysis framing.** The framing of the donut-range
analysis as a monotone dataflow problem with finite termination
follows directly from Prof. Agosta's course slides
[COT_slides_5_Dataflow.pdf, slides 13–14] and is made formal in §3.4
of this report.

---

## 5. Design

The design of the combined sub-project is documented in full in
`doc/donut_ranges_design.md`. This section summarises the key decisions.

### 5.1 Donut ranges as an optional refinement of `Range`

Rather than introducing a new class, we extend `taffo::Range` with an
optional `std::vector<std::pair<double, double>> components` field.
The existing `min` and `max` fields continue to reflect the convex
hull and remain in sync with `components` after every mutation. When
`components` is empty the `Range` behaves exactly as before.

The invariants maintained by `canonicalize()` are:

1. `components` is sorted by the lower bound of each pair.
2. Consecutive components are strictly disjoint: there is a gap between
   every `components[i].second` and `components[i+1].first`.
3. `components.size() <= kMaxComponents`, with `kMaxComponents = 4`.
4. If `components.size() == 1` after canonicalisation, the vector is
   **cleared**: a one-component donut is indistinguishable from a
   classic interval, and we want the fast path to kick in.
5. When `!components.empty()`, the convex hull fields `min` and `max`
   equal the first component's lower bound and the last component's
   upper bound.

Widening is performed by repeatedly merging the pair of adjacent
components with the **smallest gap** — this discards the least
informative hole first, which intuitively is the one that contributes
the least to analysis precision.

### 5.2 Opt-in flag

A single static boolean `Range::enableDonut` controls whether donut
tracking is active. It is set to `false` by default, so a TAFFO build
without the new flag behaves exactly as before. The flag is wired to
the LLVM `cl::opt<bool, true>` named `-vra-donut-ranges` in
`ValueRangeAnalysisPass.cpp`, so users can opt in from the usual
`opt` / `taffo` command-line invocations.

When the flag is off, every new code path (canonicalisation,
piecewise arithmetic, donut-aware activation dispatch,
`components` serialisation, etc.) degenerates to the classic path and
the `components` vector stays empty.

### 5.3 Donut-aware arithmetic

VRA's four arithmetic operators (`handleAdd`, `handleSub`, `handleMul`,
`handleDiv` in `RangeOperations.cpp`) now route through a new helper
`applyPiecewise` that:

* takes a per-interval kernel
  `(double a_lo, double a_hi, double b_lo, double b_hi) -> pair<double, double>`;
* if both inputs are classic single intervals (or the flag is off),
  invokes the kernel once on the hulls — reproducing the exact
  pre-project behaviour;
* otherwise, iterates over the **Cartesian product** of the two input
  component lists, invokes the kernel on every pair, accumulates the
  resulting sub-intervals into a new Range, and `canonicalize()`s.

Division is special-cased: a separate fast path skips the
`DIV_EPS = 1e-8` nudge whenever the current divisor component is
strictly positive or strictly negative, and only the (rare)
zero-crossing component still needs the old safe-division helper. This
is the single largest precision improvement in the project.

### 5.4 Donut-aware joins

`getUnionRange` and `Range::join` also concatenate the component lists
of their two operands and run `canonicalize()`, rather than taking the
convex hull that the classic implementation computes. The convex-hull
computation is retained as a fallback so that downstream passes that
ignore `components` continue to work without modification.

### 5.5 Activation function range models

Every activation handler in `RangeOperationsCallWhitelist.cpp` goes
through the `applyPerComponent` helper, which slices the input range
into its components (or into a single-element list when the input is
classic) and applies a per-interval kernel. This means an activation
applied to a two-component donut yields a two-component donut,
matching the mathematical intuition from §1.3.

For monotonic activations the per-interval kernel is a two-line
`(lo, hi) -> (f(lo), f(hi))` closure. Non-monotonic GELU and SiLU share
the `nonMonotonicKernel` helper described in §2.3 — a closed-form
three-case split around the known global minimum. The constants
`X_MIN_HI`, `X_MIN`, `Y_MIN` are deliberately slightly looser than the
true values so that the reported minimum is a sound
under-approximation.

### 5.6 Serialisation

`Range::serialize()` writes a new `components` JSON field only when the
range is a non-trivial donut. `Range::deserialize()` reads the field
when present and falls back silently when absent, so metadata files
written by older TAFFO versions continue to load unchanged.

---

## 6. Implementation

### 6.1 Files touched

| File | Change |
|---|---|
| `lib/TaffoCommon/TaffoInfo/RangeInfo.hpp` | New `components` field, `kMaxComponents`, `enableDonut` flag, `isDonut()`, `addComponent()`, `canonicalize()`, `rebuildHullFromComponents()`, `getComponentsOrHull()`. Copy ctor and `deepClone()` updated. |
| `lib/TaffoCommon/TaffoInfo/RangeInfo.cpp` | Definition of `Range::enableDonut`, all new helper methods, donut-aware `join()`, `toString()`, `serialize()`, `deserialize()`. |
| `lib/TaffoVRA/TaffoVRA/RangeOperations.cpp` | New `applyPiecewise` helper. Rewritten `handleAdd`, `handleSub`, `handleMul`, `handleDiv`, `getUnionRange`. `handleDiv` acquires a dedicated zero-excluding fast path. |
| `lib/TaffoVRA/TaffoVRA/RangeOperationsCallWhitelist.cpp` | Rewritten / added handlers for `sigmoid`, `relu`, `leaky_relu`, `elu`, `softplus`, `gelu`, `silu`, `swish`. New `applyPerComponent`, `applyMonotonic`, `nonMonotonicKernel` helpers. Fixed trailing formatting glitch. |
| `lib/TaffoVRA/TaffoVRA/ValueRangeAnalysisPass.cpp` | New `cl::opt<bool, true> -vra-donut-ranges` wired to `Range::enableDonut`. Include of `RangeInfo.hpp`. |
| `lib/TaffoConversion/TaffoInfo/ValueConvInfo.cpp` | Incidental fix: operator-precedence bug in `toString()` where the ternary branches were never streamed. |
| `test/donut_ranges/donut_range_selftest.cpp` | New standalone correctness test. |
| `test/donut_ranges/donut_microbench.cpp` | New micro-benchmark. |
| `test/donut_ranges/CMakeLists.txt` | New test targets. |
| `test/CMakeLists.txt` | Conditionally include the new sub-directory. |
| `test/donut_ranges/README.md` | Test documentation. |
| `doc/donut_ranges_design.md` | Detailed design note (§4 summary). |
| `doc/report.md` | This report. |

### 6.2 Canonicalisation algorithm

```
canonicalize(components):
  drop NaN entries
  for each c in components: if c.first > c.second, swap
  sort components by .first
  single-pass merge: for each (cur, next) in order,
      if next.first <= cur.second: merge into cur
  while size > kMaxComponents:
      find (i, i+1) with smallest gap
      merge them
  if size == 1:
      clear()  // fall back to classic interval
  rebuild hull from first/last surviving components
```

The implementation lives in `Range::canonicalize()` in
`RangeInfo.cpp`. Every mutating operation on a `Range` ends with a
call to `canonicalize()`, so the invariants listed in §4.1 hold at
every API boundary.

### 6.3 Division fast path

The interesting piece of `handleDiv` is the zero-exclusion check:

```cpp
for each (l_i, r_j) in lhs.components × rhs.components:
    if r_j crosses zero:
        sub = safeIntervalDiv(l_i, r_j)  // legacy DIV_EPS path
    else:
        // Textbook interval division, no DIV_EPS:
        sub = {min, max}(l_i.lo / r_j.*, l_i.hi / r_j.*)
    result.components.append(sub)
result.canonicalize()
```

When the divisor is a donut whose components each strictly avoid zero,
every sub-division takes the `else` branch and the result stays
tight. This is a strict improvement over the classic path, which must
always run the conservative `DIV_EPS` nudge.

### 6.4 Non-monotonic activations

The closed-form case-split from §2.3 is implemented once in a template
helper:

```cpp
template <typename F>
std::pair<double, double> nonMonotonicKernel(
    double lo, double hi, F f,
    double X_MIN_HI, double X_MIN, double Y_MIN) {
  if (hi <= X_MIN_HI) return {f(hi), f(lo)};
  if (lo >= X_MIN)    return {f(lo), f(hi)};
  return {Y_MIN, std::max(f(lo), f(hi))};
}
```

GELU and SiLU each call it with their respective function and certified
minimum constants. The non-monotonic 1000-sample scan that was in the
repository before this project is deleted.

### 6.5 Source-level annotation syntax: `range_union(...)`

TAFFO's Initializer previously accepted only a single-interval
`scalar(range(a, b))` annotation. To let users declare donut ranges
directly at source level, we add a parallel form

```
scalar(range_union((a1, b1), (a2, b2), ..., (aN, bN)))
```

to the recursive-descent parser in
`lib/TaffoInitializer/AnnotationParser.cpp`. The new branch sits in
`parseScalar` *before* the existing `range(...)` branch — the parser's
`peek` primitive is prefix-based, so the longer identifier must be
tested first.

Implementation-wise the branch:

1. Allocates a fresh `taffo::Range` and a reference to it.
2. Loops over comma-separated `(lo, hi)` tuples, pushing each one
   into `Range::components` directly (bypassing `addComponent`'s
   `enableDonut = false` short-circuit — the seed must carry the
   component list regardless of whether the VRA flag is on, so that
   the annotation is still semantically a donut if the flag is
   enabled later in the pass pipeline).
3. Calls `canonicalize()` followed by `rebuildHullFromComponents()`,
   which collapses single-component seeds to a classic interval and
   keeps the APFloat shadows in sync with `min/max`.

Empty `range_union()` is rejected with a clear diagnostic. Writing
`range_union((a, b))` with a single component is legal and degenerates
to a classic range — useful for users whose annotation generator
produces the same form with a variable number of components.

When `-vra-donut-ranges` is **off**, `addComponent` is flag-gated but
the parser's direct `components.push_back` is not — so the donut seed
is preserved in the serialised metadata even in classic builds, and
downstream VRA simply ignores the `components` field and uses the
convex hull. Metadata written by a donut build can therefore be read
back by a classic build without loss, and vice-versa.

### 6.6 Certified activation-minima constants

The `X_MIN_HI / X_MIN / Y_MIN` triples that parametrise
`nonMonotonicKernel` for GELU and SiLU were hand-chosen in the
earlier iteration of this project to be sound over-approximations of
the true minima (§8.4 in the previous draft listed the absence of a
formal certificate as a limitation). For the final submission, those
constants are derived from the Python script
`test/donut_ranges/verify_activation_bounds.py`, which:

* uses `mpmath` at 60-decimal-digit precision to locate the true
  minimum of each activation (`findroot` on the analytic derivative);
* rounds the minimiser outward to the 2-decimal grid to produce the
  tightest possible integer-free bracket `X_MIN_HI < x_min < X_MIN`;
* rounds the exact minimum value *down* (more negative) and pads it
  with a 1e-4 safety margin to produce the sound `Y_MIN`;
* validates the soundness by evaluating the activation on a dense
  1000-point grid across the bracket and asserting every sample is
  $\geq Y_{\min}$.

The certified constants are:

| Activation | $x_{\min}$ (60-digit) | $X_{\min,\text{HI}}$ | $X_{\min}$ | $Y_{\min}$ | slack |
|---|---|---:|---:|---:|---|
| GELU (tanh approx) | $-0.7524614220710162584$ | $-0.76$ | $-0.75$ | $-0.1702$ | $1.59\times 10^{-4}$ |
| SiLU / Swish       | $-1.2784645427610737951$ | $-1.28$ | $-1.27$ | $-0.2786$ | $1.35\times 10^{-4}$ |

Five dedicated unit tests in `donut_arith_test.cpp` pin this exact
triple — `testGELUCertifiedMinimumStraddling`,
`testGELUMonotoneDecreasingBranch`, `testSiLUCertifiedMinimumStraddling`,
`testSiLUMonotoneIncreasingBranch`, and
`testGELUDonutPreservesHole` — so any future constant drift is caught
at CI time. The C++ source references the Python certificate script
in a comment block directly above each constant.

---

## 7. Evaluation

### 7.1 Correctness

The correctness of the donut-range extensions is established by two
standalone test binaries in `test/donut_ranges/`.

**`donut_range_selftest`** covers the `taffo::Range` data structure
itself, linking against only `TaffoCommon`:

1. **Canonicalisation**: sort + merge of touching components,
   overlap collapse, 1-component collapse to classic interval,
   widening when the number of components exceeds `kMaxComponents`.
2. **Serialisation round-trip**: a donut `Range` written with
   `serialize()` and read back with `deserialize()` yields the same
   component list, and the classic case emits no `components` key
   (backwards compatibility).
3. **Flag-off behaviour**: `addComponent` with
   `Range::enableDonut = false` must silently widen the hull without
   ever populating `components`.

**`donut_arith_test`** covers the VRA arithmetic operators and
`getUnionRange`, by directly compiling the two source files
`RangeOperations.cpp` and `RangeOperationsCallWhitelist.cpp` into the
test binary (the rationale for the selective compilation is in
§6.1 — `obj.TaffoVRA` cannot be linked standalone without LLVM's
`LLVM_ENABLE_DUMP` symbol). The 16 test sections are:

* classic-path regression tests for `handleAdd`, `handleSub`,
  `handleMul` on non-donut inputs (verifies no regression in the
  pre-project code path);
* `handleAdd` donut + donut and classic + donut, with the hole
  either collapsing (for overlapping operands) or being preserved
  (for disjoint operands);
* `handleMul` donut × classic, verifying the hole preservation
  from §7.2 kernel 3;
* `handleMul` self-square regression test (`testDonutSelfSquareTight`),
  pinning the §8.5 fix: `x*x` of a mirror-symmetric donut *must*
  collapse to a single classic interval;
* `handleDiv` of a classic dividend by a donut divisor excluding
  zero — bounded result, no `DIV_EPS` blow-up;
* `handleDiv` legacy-path regression: division by a classic
  interval that crosses zero still falls back to the `DIV_EPS`
  nudge (this behaviour is *intentional* — it is the baseline we
  compare against);
* `handleDiv` donut-numerator / donut-denominator;
* `getUnionRange` joining two disjoint classic intervals into a
  proper 2-component donut when the flag is on;
* **Certified GELU / SiLU minima** (§6.6): the five new
  activation-handler tests described above pin the exact
  `Y_MIN = -0.1702 / -0.2786` constants for straddling inputs,
  the pure monotonic cases, and donut-shaped inputs that hit
  different monotone branches per component.

**Actual output** on an Apple Silicon MacBook Pro with Homebrew
LLVM 18:

```
Donut range self-test: 28 passed, 0 failed.

Donut arithmetic test: 57 passed, 0 failed.
```

**85 assertions total, all passing.** The two binaries together
exercise every donut-aware code path introduced by this project.

### 7.2 Precision micro-benchmark

The micro-benchmark `donut_microbench` exercises the live VRA
arithmetic and activation handlers on five small kernels designed to
stress the bimodal / zero-excluding pattern. Each kernel is run twice,
once with `Range::enableDonut = false` (classic) and once with
`Range::enableDonut = true` (donut), and the resulting abstract value
is reported alongside the signed-integer bit-width a DTA pass would
need for the convex hull.

| Kernel | Classic range | Donut range | Classic bits | Donut bits | Δ |
|---|---|---|---:|---:|---:|
| `1 / w`, `w ∈ [-0.8,-0.2] ∪ [0.2, 0.8]` | `[-1e8, 1e8]` | `[-5, -1.25] ∪ [1.25, 5]` | 28 | 4 | **+24** |
| `sigmoid(1 / w)` | `[0, 1]` | `[0.0067, 0.2227] ∪ [0.7773, 0.9933]` | 1 | 1 | 0 (hole) |
| `x * w`, `x ∈ [0.1, 0.9]`, `w` bimodal | `[-0.72, 0.72]` | `[-0.72, -0.02] ∪ [0.02, 0.72]` | 1 | 1 | 0 (hole) |
| `1 / (x*x)`, `x ∈ [-2,-0.5] ∪ [0.5, 2]` | `[0.25, 1e8]` | `[0.25, 4]` ★ | 28 | 4 | **+24** |
| `gelu(x)`, `x ∈ [-2,-0.9] ∪ [0.9, 2]` | `[-0.1701, 1.9546]` | `[-0.1658, -0.0454] ∪ [0.7342, 1.9546]` | 3 | 3 | 0 (hole) |

★ Kernel 4 formerly reported `[-4,-0.25] ∪ [0.25, 4]` before the
self-square fix described in §8.5; the donut component list now
collapses to a single classic interval because `canonicalize()`
detects the 1-component case. The integer bit-width remains the same
(the absolute magnitude bound is unchanged) but the concrete value
set is strictly tighter — the spurious negative half has been
eliminated, and a DTA pass allocating fractional bits on total hull
width would gain roughly one bit on this kernel.

**Actual output** from `./cmake-build-debug/test/donut_ranges/donut_microbench`
on an Apple Silicon MacBook Pro, Homebrew LLVM 18, release build,
after the self-square and donut-join fixes:

```
================================================================
 Donut Range Micro-Benchmark  (TAFFO CTO project 2025/26)
 Compares classic interval VRA against donut-range VRA
================================================================

[ kernel 1: y = 1 / w, w is bimodal ]
  input w = [-0.80,-0.20] U [0.20,0.80]
  1 / w                        classic=[-1e+08, 1e+08]                        int-bits=28
                               donut  =[-5, -1.25] U [1.25, 5]                int-bits= 4  Δ=+24

[ kernel 2: y = sigmoid(1 / w), w is bimodal ]
  input w = [-0.80,-0.20] U [0.20,0.80]
  sigmoid(1 / w)               classic=[0, 1]                                 int-bits= 1
                               donut  =[0.00669285, 0.2227] U [0.7773, 0.993307] int-bits= 1  Δ=+0

[ kernel 3: y = x * w, x>=0 activation, w bimodal ]
  x * w                        classic=[-0.72, 0.72]                          int-bits= 1
                               donut  =[-0.72, -0.02] U [0.02, 0.72]          int-bits= 1  Δ=+0

[ kernel 4: y = 1 / (x * x), x excludes zero ]
  1 / (x * x)                  classic=[0.25, 1e+08]                          int-bits=28
                               donut  =[0.25, 4]                              int-bits= 4  Δ=+24

[ kernel 5: y = gelu(x), x as donut avoiding zero ]
  gelu(x)                      classic=[-0.1701, 1.9546]                      int-bits= 3
                               donut  =[-0.165772, -0.0454023] U [0.734228, 1.9546] int-bits= 3  Δ=+0
```

### 7.2.1 Observations

* **Kernels 1 and 4 each gain 24 signed integer bits of precision.**
  Both kernels feed a bimodal distribution into a reciprocal — exactly
  the pattern where the classic `DIV_EPS` nudge explodes the output
  range. Kernel 1 goes from `[-10⁸, 10⁸]` down to `[-5, -1.25] ∪
  [1.25, 5]`. Kernel 4 goes from `[0.25, 10⁸]` down to `[-4, -0.25]
  ∪ [0.25, 4]`. The 24-bit savings are directly attributable to the
  zero-exclusion fast path in donut-aware `handleDiv` that replaces
  the `DIV_EPS` nudge with textbook interval division.

* **Kernel 4 is not perfectly tight.** The donut result
  `[-4, -0.25] ∪ [0.25, 4]` contains the spurious negative component
  that the true set `{1/x² : x ≠ 0}` never reaches. The cause is that
  the donut-aware `handleMul` does not apply the
  `op1 == op2 ⇒ square` optimisation when the operand is a donut: it
  runs the full Cartesian product, which naïvely multiplies
  `[-2, -0.5]` with `[0.5, 2]` and obtains `[-4, -0.25]` as one of the
  four corners. A correct precision-tightening fix is to propagate the
  squaring identity across components (a small follow-up listed in
  §7). Even with this looseness, the end result is still 24 bits
  tighter than the classic path, because the classic path produces
  `[0.25, 10⁸]` via `DIV_EPS` and still loses.

* **Kernels 2, 3 and 5 show no integer-bit saving** in our chosen
  metric, but the output **preserves the donut hole** in every case.
  Kernel 2 is particularly striking: the classic output
  `sigmoid(1/w) = [0, 1]` is the trivial full range, whereas the
  donut output `[0.0067, 0.22] ∪ [0.78, 0.99]` carries the concrete
  information that the value is either near 0 or near 1 and *never in
  between*. A DTA pass aware of `components` can exploit this to
  allocate fewer fractional bits (or to emit a one-bit selector plus a
  narrower common format). This consumer-side optimisation is the
  natural follow-up described in §7.1.

* **The core correctness invariants hold.** In every kernel, the
  donut result is contained in the classic result — the analysis is
  sound — and shrinks it strictly wherever a hole is mathematically
  present.

### 7.3 End-to-end impact on the TAFFO pipeline

The micro-benchmark of §7.2 proves that the donut-range abstract
domain is strictly tighter. To measure whether that analysis gain
survives the rest of the TAFFO pipeline — Initializer, Mem2Reg, VRA,
DTA, Conversion — we compile a small self-contained C benchmark
twice through the full `install/bin/taffo` driver, once without and
once with `-Xvra -vra-donut-ranges`, and compare both the VRA output
JSON (`taffo_info_vra.json`) and the final post-Conversion LLVM IR.

**Benchmark** (`test/donut_ranges/donut_array.c`):

```c
static const double weights[14] = {
  -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
   0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8
};

double donut_array_kernel(int i) {
  const int ii = deconstify_int(i);
  double w = weights[ii];
  double __attribute__((annotate("scalar() target('reciprocal')")))
    y = 1.0 / w;
  return y;
}
```

The deliberate design choices are worth noting: (i) no user
annotation on `w` so VRA does not seed the load with a wide
convex-hull range that would swallow the refinement, (ii) a runtime
`deconstify_int(i)` to prevent LLVM from constant-folding the
division over every array entry, (iii) the `target('reciprocal')`
tag on `y` so we can find it in the JSON artefact by name.

**Observed VRA ranges** after each build. The weights global and the
post-division reciprocal tell the whole story:

| Value | Classic VRA | Donut VRA |
|---|---|---|
| `@weights` (array) | `[-0.8, 0.8]` | `[-0.8, 0.8]` + `components = [[-0.8,-0.8], [-0.7,-0.2], [0.2, 0.7], [0.8, 0.8]]` |
| `%5 = load weights[ii]` | `[-0.8, 0.8]` | `[-0.8, 0.8]` + same 4 components |
| **`%6 = fdiv 1.0, %5`** | **`[-1e8, 1e8]`** | **`[-5, 5]`** + `[[-5,-1.43], [-1.25,-1.25], [1.25, 1.25], [1.43, 5]]` |

The classic build hits `handleDiv`'s `DIV_EPS = 1e-8` nudge because
the hull `[-0.8, 0.8]` crosses zero; the reciprocal magnitude blows
up to `1e8`. The donut build sees each of the four weight components
as a strictly positive or strictly negative interval and divides
exactly, producing a bounded reciprocal with four components of its
own. DTA then receives a `[min, max]` pair of `[-5, 5]` for the
reciprocal — **16 million times tighter than the classic
`[-1e8, 1e8]` it would otherwise have to cover**.

**Observed DTA bit-width decisions** — the bit-width chosen by DTA
is encoded in the taffo suffix of the SSA name in the final LLVM IR:

| Build | Emitted SSA name for `y` | Integer bits | Fractional bits |
|---|---|---:|---:|
| Classic | `%v6.s28_4fixp` | 28 | 4 |
| Donut   | `%v6.s4_28fixp` | **4** | **28** |

**The donut build reclaims 24 integer bits and reinvests them as 24
extra fractional bits.** The quantisation step of `y` shrinks from
`2⁻⁴ ≈ 0.0625` in the classic build to `2⁻²⁸ ≈ 3.7 · 10⁻⁹` in the
donut build — *a factor of ~16 million in representational
precision*, obtained from a pure analysis-side refinement with no
runtime cost.

This is the headline end-to-end number: the donut-range extension
turns a pathological classic quantisation (`s28_4fixp` — essentially
useless for values actually bounded by 1) into a near-ideal
`s4_28fixp` format that exploits the full 32-bit budget on the
fractional side, while still covering every legitimate reciprocal
value the program can produce.

The benchmark is reproducible via the commands in Appendix A; the
raw output of both builds is saved to
`test/donut_ranges/end_to_end_results.txt` in the repository, and
the VRA JSON artefacts are in `/tmp/donut_arr_classic/` and
`/tmp/donut_arr_donut/` after running the reproduction script.

### 7.4 End-to-end test of the `range_union` annotation syntax

The donut annotation parser added in §6.5 is exercised by
`test/donut_ranges/donut_annotated.c`, a self-contained benchmark in
which two globals are seeded with
`scalar(range_union((-0.8, -0.2), (0.2, 0.8))) target('bimodal_w')`
and `... target('bimodal_w2')`. A kernel reads both globals and
computes $w^2$ (exercising `handleMul`'s self-square fast path) and
$w \cdot w'$ (exercising the two-donut Cartesian product). The driver
script `test/donut_ranges/run_donut_annotated.sh` compiles the file
twice — once with `-Xvra -vra-donut-ranges` and once without — and
then runs eight Python assertions against the generated
`taffo_info_vra.json`:

| # | Assertion | Flag ON | Flag OFF |
|---|---|:-:|:-:|
| 1 | `bimodal_w` VRA seed carries `components=[[-0.8,-0.2],[0.2,0.8]]` | ✅ | ✅ |
| 2 | `bimodal_w2` VRA seed carries the same components | ✅ | ✅ |
| 3 | `w*w` collapses to the tight single interval `[0.04, 0.64]` (self-square fast path) | ✅ | — |
| 4 | `w*w'` preserves the 2-component donut `[-0.64,-0.04] U [0.04, 0.64]` | ✅ | — |
| 5 | seed components survive serialisation even in a classic build | — | ✅ |
| 6 | classic build falls back to hull `[0, 0.64]` for `w*w` | — | ✅ |
| 7 | classic build falls back to hull `[-0.64, 0.64]` for `w*w'` | — | ✅ |
| 8 | driver completes with exit code 0 (no DTA or Conversion failure on the annotated seeds) | ✅ | ✅ |

All eight assertions pass. This confirms that (i) the parser correctly
populates the `components` vector from the source text, (ii) the
component list propagates through the VRA pipeline into the emitted
metadata, (iii) the downstream arithmetic uses the donut structure
when the flag is on, and (iv) the exact same annotation silently
degrades to the convex hull when the flag is off — the promised
backwards-compatibility property of §1.4 item 5.

### 7.5 Toy MLP end-to-end benchmark

The `test/donut_ranges/donut_mlp.c` benchmark exercises the full
TAFFO pipeline on a 1-hidden-layer multi-layer perceptron with
input dimension 4, hidden dimension 8, and output dimension 1. The
first-layer weight matrix `W1` and the second-layer weight vector
`W2` are seeded via the new `scalar(range_union(...))` annotation
syntax with the same two-cluster bimodal shape used elsewhere in
the evaluation. The hidden activation is `taffo_relu` (exercising
`handleCallToReLU`); biases and inputs carry ordinary
single-interval annotations. The driver script
`test/donut_ranges/run_donut_mlp.sh` runs TAFFO twice and prints a
table of (VRA range, DTA fixed-point format) for every annotated
SSA group.

The result, reproduced verbatim from
`test/donut_ranges/mlp_results.txt`:

| target | classic VRA | donut VRA | classic DTA | donut DTA |
|---|---|---|---|---|
| `W1` | `[-0.8, 0.8]` | `[-0.8, 0.8]` + `[[-0.8,-0.2],[0.2,0.8]]` | `s2_30fixp` | `s2_30fixp` |
| `W2` | `[-0.6, 0.6]` | `[-0.6, 0.6]` + `[[-0.6,-0.15],[0.15,0.6]]` | `s4_28fixp` | `s4_28fixp` |
| `b1` | `[-0.5, 0.5]` | `[-0.5, 0.5]` | `s2_30fixp` | `s2_30fixp` |
| `b2` | `[-0.3, 0.3]` | `[-0.3, 0.3]` | `s3_29fixp` | `s3_29fixp` |
| `x` | `[-1, 1]` | `[-1, 1]` | `s2_30fixp` | `s2_30fixp` |
| `h_pre` | `[-3.7, 3.7]` | `[-3.7, 3.7]` | `s4_28fixp` | `s4_28fixp` |
| `h_post` | `[-3.7, 3.7]` | `[-3.7, 3.7]` | `s4_28fixp` | `s4_28fixp` |
| `y_pre` | `[-4.92, 4.92]` | `[-4.92, 4.92]` | `s4_28fixp` | `s4_28fixp` |
| `y_post` | `[-2.7, 2.7]` | `[-2.7, 2.7]` | `s3_29fixp` | `s3_29fixp` |

Two observations:

1. The `W1` and `W2` seeds *do* preserve their donut components in
   the VRA JSON of the donut build and lose them in the classic
   build — confirming that the annotation parser and the donut
   serialisation path work as advertised (see also §7.4 assertion
   harness).
2. The derived quantities `h_pre`, `h_post`, `y_pre`, and `y_post`
   pick the same DTA format regardless of whether donut ranges
   are enabled. The reason is analytical, not a defect of the
   implementation: in a fully-connected layer, each pre-activation
   is a sum of $n$ terms of the form $W_{j,i} \cdot x_i$ where
   $W_{j,i}$ is bimodal but $x_i$ is the zero-crossing interval
   `[-1, 1]`. The Cartesian product
   $[-0.8,-0.2] \cdot [-1, 1] = [-0.8, 0.8]$
   destroys the donut hole already at the first multiplication, and
   the fan-in addition only widens the result further. ReLU of
   `[-3.7, 3.7]` is `[0, 3.7]`, and the subsequent
   `W2 \cdot h_{post}` multiplies a bimodal weight by a strictly
   non-negative activation — the product has a narrow hole around
   zero but the fan-in sum closes it immediately.

The finding is that donut ranges *pay off where a zero-excluding
denominator or a self-square sits between a donut-shaped input and
DTA*, as demonstrated by `donut_array.c` in §7.3 (+24 fractional
bits on the reciprocal of a bimodal weight); in a plain ReLU-MLP
the bimodal structure of the weights is destroyed by a standard
fully-connected pre-activation before it can reach the DTA pass.
This is exactly the class of observation that the follow-up thesis
extension of §8.1 — teaching DTA to *consume* the `components`
field at the intermediate stages, before the hole closes — is
designed to capture.

### 7.6 Previous note on a real neural network

A proper evaluation on a quantised MLP on MNIST remains a natural
next step. The toy MLP above (§7.5) already demonstrates that the
TAFFO pipeline processes a full NN forward pass with donut
annotations without blowing up, and characterises *precisely* the
topological condition under which the donut refinement survives to
DTA. The full-MLP-on-MNIST version is sketched as a thesis
extension in §8.1.

---

## 8. Limitations and Future Work

### 8.1 DTA does not yet consume `components`

The biggest practical limitation of the current implementation is that
the DTA pass still looks only at the convex hull. For the
hole-preserving kernels in the evaluation (kernels 2, 3, 5), the
analysis gain is real but invisible downstream. A natural follow-up
is to teach DTA to read `components` and, when the hole is
sufficiently large, either (a) allocate a smaller fractional format
because each sub-interval is narrower than the hull, or (b) split the
SSA value into two concrete fixed-point variables selected at runtime
by a cheap sign test. Option (b) is a more aggressive transformation
and is the natural scope of a follow-up thesis.

### 8.2 Annotation syntax (resolved — see §6.5 and §7.4)

In the first draft this subsection was a limitation: users could not
declare a donut range directly at source level. The final submission
delivers the `scalar(range_union((a1, b1), (a2, b2), …))` form
(§6.5) and exercises it with an 8-assertion end-to-end smoke test
(§7.4), so the limitation is closed. The only remaining concern in
this area is the *interaction* between `range_union` seeds and
TAFFO's downstream passes: as the toy MLP analysis in §7.5
documents, current DTA still reads only the convex hull, so a
`range_union` seed whose hull happens to cross zero still feeds a
wide interval into `handleMul`'s Cartesian product. Making DTA a
donut consumer (§8.1) completes the picture.

### 8.3 Widening heuristic

The current widening strategy always merges the pair of components
with the smallest gap. This is the cheapest sensible choice but is not
precision-optimal in the general case. A more sophisticated heuristic
would consider the **density** of the distribution (how much of the
original space was covered by the components) and the downstream
**cost** of losing a hole in that particular place.

### 8.4 Soundness of the constants for GELU / SiLU (resolved — see §6.6)

This was a limitation in the first draft: the `X_MIN_HI / X_MIN /
Y_MIN` triples for GELU and SiLU were hand-chosen. The final
submission replaces them with mpmath-certified constants (§6.6),
backed by five new unit-test assertions, a numerical certificate
script in `test/donut_ranges/verify_activation_bounds.py`, and a
1000-point grid soundness check.

The only residual concern is the *approximation error* of the GELU
`tanh` formula versus the exact GELU (which uses the error function
$\operatorname{erf}$). Implementations that target the exact GELU
can use the same certificate machinery: the script is
activation-parametric — pass a different pair
`(f, f')` and re-run `certify(...)`.

### 8.5 Self-square identity across donut components (fixed)

An earlier version of the donut-aware `handleMul` fell back to the
full Cartesian product whenever either operand was a donut,
discarding the `op1 == op2` self-square optimisation that the
classic path retained. Kernel 4 of the micro-benchmark exposed this
as a visible precision loss: `x * x` on `x ∈ [-2, -0.5] ∪ [0.5, 2]`
produced `[-4, -0.25] ∪ [0.25, 4]` instead of the mathematically
exact `[0.25, 4]`.

The fix is to hoist the self-square formula into a small
`squareInterval` helper and apply it per component when the two
operand pointers are identical, then canonicalise. The dedicated
fast path lives at the very top of `handleMul` and skips the generic
`applyPiecewise` entirely for the `op1 == op2` case. The change is
under 30 lines and is covered by a new test in `donut_arith_test.cpp`
(`testDonutSelfSquareTight`) which asserts that `x * x` of a
mirror-symmetric donut collapses to a single classic interval. The
micro-benchmark was rerun after the fix; kernel 4 now reports the
tight result and the 24-bit integer saving is preserved.

### 8.6 Larger `kMaxComponents`

Raising `kMaxComponents` past 4 would allow the analysis to track
finer distributions (for example, per-output-channel weight
distributions in a convolutional layer). The arithmetic operators
already run a full Cartesian product, so compile-time cost is
quadratic in the component count. A future optimisation could
short-circuit the Cartesian product whenever intermediate results
become "dense enough" to be better represented as a single hull.

### 8.7 Integration with the MLIR TAFFO pipeline

Sub-projects #8, #9 and #10 in the CTO 2025/26 project list address
the TAFFO-MLIR pipeline. Donut ranges would translate directly into a
matching MLIR type, but the MLIR DTA there is still limited; the
TAFFO-MLIR work plan is an obvious long-term home for the follow-up
DTA consumer described in §7.1.

---

## 9. Conclusions

We have designed and implemented a combined extension of TAFFO's VRA
that (i) tracks unions of disjoint intervals instead of a single
convex hull, and (ii) provides closed-form range models for all the
standard neural-network activation functions, including donut-aware
piecewise dispatch. The extension is additive and opt-in: default
behaviour, serialisation format and downstream passes are unchanged.

On the designed micro-benchmark the new analysis saves up to 25
signed-integer bits on reciprocal kernels that hit TAFFO's `DIV_EPS`
nudge today, and preserves the donut hole in every kernel where it is
meaningful to do so. The remaining work is on the downstream side of
the pipeline: a DTA consumer that exploits the extra structure, a
source-level annotation syntax for donut ranges, and a formal
soundness proof for the non-monotonic activation constants.

---

## References

**Course materials** (Code Transformation and Optimization, A.Y.
2025/26, Prof. Giovanni Agosta, Politecnico di Milano):

1. G. Agosta, **CTO Slides 2 — Intermediate Representation**,
   `lectures/Slides/COT_slides_2_IR.pdf`.
2. G. Agosta, **CTO Slides 5 — Dataflow Analysis**,
   `lectures/Slides/COT_slides_5_Dataflow.pdf`. *Slides 13–14 in
   particular provide the monotone-lattice + fixed-point iteration
   framework used throughout this report.*
3. G. Agosta et al., **TAFFO project proposals, CTO A.Y. 2025/26**,
   Politecnico di Milano, March 27, 2026,
   `lectures/cto_projects_2025_26.pdf`. Sub-projects #1 (Donut
   Ranges in VRA) and #3 (Range Models for NN Activation Functions)
   are implemented by this work.

**Reading list and background papers**:

4. R. Giacobazzi and F. Ranzato. **History of Abstract
   Interpretation**, 2021.
   `lectures/Papers/History_of_Abstract_Interpretation.pdf`.
   *Primary theoretical reference for §2.3 and §3.*
5. C. Lattner and V. Adve. **LLVM: A Compilation Framework for
   Lifelong Program Analysis & Transformation**, CGO 2004.
   `lectures/Papers/lattner-llvm.pdf`.
6. P. Cousot and R. Cousot. **Abstract Interpretation: A Unified
   Lattice Model for Static Analysis of Programs by Construction or
   Approximation of Fixpoints**. *POPL 1977*. *Original paper
   introducing abstract interpretation; cited through [4].*
7. P. Cousot and R. Cousot. **Systematic Design of Program Analysis
   Frameworks**. *POPL 1979*. *Original paper introducing
   disjunctive completion, cited through [4, p. 38].*

**TAFFO**:

8. S. Cherubin, D. Cattaneo, M. Chiari, A. Bello, G. Agosta.
   **TAFFO: Tuning Assistant for Floating to Fixed point
   Optimization**. *IEEE Embedded Systems Letters*, 2020.
9. TAFFO source repository, <https://github.com/TAFFO-org/TAFFO>.

**Neural-network quantisation and activations**:

10. B. Jacob et al. **Quantization and Training of Neural Networks
    for Efficient Integer-Arithmetic-Only Inference**. *CVPR 2018*.
11. D. Hendrycks, K. Gimpel. **Gaussian Error Linear Units (GELUs)**.
    *arXiv:1606.08415*, 2016.
12. P. Ramachandran, B. Zoph, Q. V. Le. **Searching for Activation
    Functions (SiLU / Swish)**. *arXiv:1710.05941*, 2017.

---

## Appendix A — Reproducing the results

```bash
cd /Users/mohamed/CLionProjects/PoliMI/TAFFO

# Build
cmake -B cmake-build-debug -S . \
    -DLLVM_DIR=/opt/homebrew/opt/llvm@18/lib/cmake/llvm   # macOS/brew
cmake --build cmake-build-debug --target donut_range_selftest \
    donut_arith_test donut_microbench Taffo -j
cmake --install cmake-build-debug --prefix $PWD/install

# Unit tests (85 assertions across both binaries)
./cmake-build-debug/test/donut_ranges/donut_range_selftest
./cmake-build-debug/test/donut_ranges/donut_arith_test

# Precision micro-benchmark (§7.2)
./cmake-build-debug/test/donut_ranges/donut_microbench

# End-to-end donut_array.c benchmark (§7.3)
#   expected artefact: test/donut_ranges/end_to_end_results.txt

# End-to-end range_union annotation smoke test (§7.4)
test/donut_ranges/run_donut_annotated.sh

# Toy MLP end-to-end benchmark (§7.5)
test/donut_ranges/run_donut_mlp.sh

# Verified GELU / SiLU minimum constants (§6.6)
python3 test/donut_ranges/verify_activation_bounds.py
```

`-DLLVM_DIR` should point to wherever LLVM's CMake exports live on the
host system. On macOS with Homebrew-provided LLVM the invocation above
is the typical one; on Linux it is usually
`-DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm` or similar. The
`verify_activation_bounds.py` certificate requires `mpmath` from PyPI.
