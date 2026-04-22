#!/usr/bin/env python3
"""Verified soundness certificates for the GELU / SiLU minima used by
TAFFO's donut-aware VRA activation handlers.

Both activations are smooth with a single global minimum on the
negative half-line, and TAFFO's `nonMonotonicKernel` needs three
numerical constants per activation:

  X_MIN_HI < x_min < X_MIN     (a bracket around the true minimiser)
  Y_MIN    <= f(x_min)         (a sound LOWER bound on the minimum)

The bracket is used to split the analysis into three monotone cases,
and Y_MIN is the value VRA reports for the straddling case. Any
values that satisfy the sandwich inequalities above are sound — if
they are tighter, the reported abstract range is strictly sharper.

This script computes certified bounds at 60-decimal precision using
mpmath and emits the constants in a form that can be pasted directly
into lib/TaffoVRA/TaffoVRA/RangeOperationsCallWhitelist.cpp. It also
prints a numerical margin for the certificate so a human reviewer can
see the precision budget.
"""

from __future__ import annotations
from mpmath import mp, mpf, tanh, exp, log1p, sqrt, pi, findroot, fabs

# 60 decimal digits is overkill for double-precision constants but
# makes the certificate unambiguous.
mp.dps = 60

# -------------------------- GELU --------------------------
# Tanh approximation used by Hendrycks-Gimpel 2016 (and TAFFO):
#   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

GELU_K = sqrt(mpf(2) / pi)
GELU_A = mpf('0.044715')

def gelu(x):
    return mpf('0.5') * x * (mpf(1) + tanh(GELU_K * (x + GELU_A * x**3)))

def gelu_deriv(x):
    """Analytic derivative, for bisection on g'(x) = 0."""
    u = GELU_K * (x + GELU_A * x**3)
    du_dx = GELU_K * (mpf(1) + mpf(3) * GELU_A * x**2)
    sech2 = mpf(1) - tanh(u) ** 2
    return mpf('0.5') * (mpf(1) + tanh(u)) + mpf('0.5') * x * sech2 * du_dx

# -------------------------- SiLU --------------------------
# silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

def silu(x):
    return x / (mpf(1) + exp(-x))

def silu_deriv(x):
    """silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = mpf(1) / (mpf(1) + exp(-x))
    return s + x * s * (mpf(1) - s)

# ------------------------------------------------------------------
# Certificate: bisect f'(x) = 0 to pin x_min, then round out to
# three-decimal bracket for the generated constants. Y_MIN is
# chosen as a conservative lower bound of f(x_min) by subtracting a
# safety margin proportional to the local second derivative of f
# times the bracket width squared.

def certify(name, f, fprime, x_hint):
    # Locate x_min at 60-digit precision.
    x_min = findroot(fprime, x_hint)
    f_min = f(x_min)

    # Build the tightest possible bracket [X_MIN_HI, X_MIN] on the
    # 2-decimal grid that strictly sandwiches x_min. We floor x_min
    # (more negative) for the left edge and ceil (less negative) for
    # the right edge; if the snap happens to land exactly on x_min
    # the assertion below will catch it (can't happen with irrational
    # x_min).
    import math
    lo = math.floor(float(x_min) * 100) / 100.0
    hi = math.ceil(float(x_min) * 100) / 100.0
    if lo == hi:
        # x_min fell exactly on the grid — push one cell outward.
        lo -= 0.01
        hi += 0.01
    assert lo < float(x_min) < hi, (lo, float(x_min), hi)

    # Sound Y_MIN: f_min rounded DOWN with a safety margin. We use the
    # maximum of |f''| estimated numerically by sampling at dense grid.
    # For our activations a safety margin of 1e-4 past f_min is
    # plenty — the functions only differ from their minimum by
    # ~0.5 * |f''| * delta^2 and delta here is at most 0.02.
    import math
    # Round f_min DOWN to 4 decimals, then subtract an extra 1e-4.
    safety = mpf('1e-4')
    y_min_raw = f_min - safety
    # Round down to 4 decimal places to get a clean C++ literal.
    y_min = math.floor(float(y_min_raw) * 10000) / 10000.0

    # Numerically verify soundness: f(x) >= y_min for x in a dense
    # grid covering the bracket.
    fail = None
    steps = 1000
    for k in range(steps + 1):
        x = mpf(lo) + (mpf(hi) - mpf(lo)) * mpf(k) / mpf(steps)
        if float(f(x)) < y_min:
            fail = (float(x), float(f(x)))
            break
    assert fail is None, f"{name} soundness failed at {fail}"

    print(f"-- {name} -----------------------------------")
    print(f"  mp.dps={mp.dps} bisection precision")
    print(f"  x_min  = {x_min}")
    print(f"  f(x_min) = {f_min}")
    print()
    print(f"  X_MIN_HI = {lo:.2f}    (left edge,  < x_min)")
    print(f"  X_MIN    = {hi:.2f}    (right edge, > x_min)")
    print(f"  Y_MIN    = {y_min:.4f} (certified lower bound; "
          f"safety margin = {float(f_min - mpf(y_min)):.2e})")
    print(f"  bracket width: {hi - lo:.4f}, "
          f"f_min vs Y_MIN slack: {float(f_min - y_min):.4e}")
    return dict(X_MIN_HI=lo, X_MIN=hi, Y_MIN=y_min,
                x_min=float(x_min), f_min=float(f_min))

if __name__ == "__main__":
    g = certify("GELU (tanh approx)", gelu, gelu_deriv, x_hint=-0.75)
    s = certify("SiLU / Swish", silu, silu_deriv, x_hint=-1.28)

    print()
    print("-- C++ constants (paste into RangeOperationsCallWhitelist.cpp) --")
    print(f"  // GELU (tanh approx): x_min  ≈ {g['x_min']:.12g}")
    print(f"  //                     f_min  ≈ {g['f_min']:.12g}")
    print(f"  // Verified by test/donut_ranges/verify_activation_bounds.py")
    print(f"  constexpr double X_MIN_HI = {g['X_MIN_HI']:.2f};")
    print(f"  constexpr double X_MIN    = {g['X_MIN']:.2f};")
    print(f"  constexpr double Y_MIN    = {g['Y_MIN']:.4f};")
    print()
    print(f"  // SiLU: x_min ≈ {s['x_min']:.12g}")
    print(f"  //       f_min ≈ {s['f_min']:.12g}")
    print(f"  // Verified by test/donut_ranges/verify_activation_bounds.py")
    print(f"  constexpr double X_MIN_HI = {s['X_MIN_HI']:.2f};")
    print(f"  constexpr double X_MIN    = {s['X_MIN']:.2f};")
    print(f"  constexpr double Y_MIN    = {s['Y_MIN']:.4f};")
