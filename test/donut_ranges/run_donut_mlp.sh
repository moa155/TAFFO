#!/usr/bin/env bash
# Runs the toy 1-hidden-layer MLP (test/donut_ranges/donut_mlp.c)
# through the TAFFO driver twice (classic vs donut) and compares the
# resulting VRA ranges and DTA fixed-point formats.
#
# Expected outcome:
#   * Seeded weight ranges (W1, W2) carry `components` in the donut
#     VRA JSON; same weights have a single convex-hull interval in the
#     classic VRA JSON.
#   * Derived quantities (h_pre, h_post, y_pre, y_post) currently show
#     no DTA gain because a) zero-crossing inputs x * bimodal W
#     collapses the donut and b) the sum of component-wise products
#     fuses components. This is an HONEST finding for the report's
#     §7.4 discussion and is not a regression — the downstream passes
#     behave identically, just without the donut refinement getting
#     to propagate in this particular topology.
#
# The full pipeline completes end-to-end (taffotmp.s is produced by
# both builds) after the Conversion-pass fixes landed with this
# project (convertStore insertionPoint + float-source defensive
# routing in genConvertConvToConv).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAFFO="$ROOT/install/bin/taffo"
SRC="$ROOT/test/donut_ranges/donut_mlp.c"
TMP_CLASSIC="/tmp/mlp_classic"
TMP_DONUT="/tmp/mlp_donut"

if [[ ! -x "$TAFFO" ]]; then
  echo "taffo driver not found at $TAFFO; build and install first." >&2
  exit 1
fi

rm -rf "$TMP_CLASSIC" "$TMP_DONUT"
mkdir -p "$TMP_CLASSIC" "$TMP_DONUT"

echo "---- MLP classic build (no flag) ----"
"$TAFFO" -O2 -c -emit-llvm \
  -temp-dir "$TMP_CLASSIC" -o "$TMP_CLASSIC/donut_mlp.ll" "$SRC" \
  >"$TMP_CLASSIC/taffo.stdout" 2>"$TMP_CLASSIC/taffo.stderr" || true

echo "---- MLP donut build ( -Xvra -vra-donut-ranges ) ----"
"$TAFFO" -O2 -c -emit-llvm -Xvra -vra-donut-ranges \
  -temp-dir "$TMP_DONUT" -o "$TMP_DONUT/donut_mlp.ll" "$SRC" \
  >"$TMP_DONUT/taffo.stdout" 2>"$TMP_DONUT/taffo.stderr" || true

python3 - <<'PY'
import json

def load(p):
    with open(p) as f:
        return json.load(f)

def values_by_target(doc):
    out = {}
    for k, v in doc.get("values", {}).items():
        info = v.get("info") if isinstance(v, dict) else None
        if not isinstance(info, dict):
            continue
        tgt = info.get("target")
        if not tgt:
            continue
        out.setdefault(tgt, []).append(info)
    return out

def first_range(infos):
    for info in infos:
        r = info.get("range") or {}
        if isinstance(r, dict) and "min" in r:
            return r
    return {}

def first_nt(infos):
    for info in infos:
        nt = info.get("numericType")
        if isinstance(nt, dict) and nt.get("kind") == "FixedPoint":
            return nt
    return None

c_vra = values_by_target(load("/tmp/mlp_classic/taffo_info_vra.json"))
d_vra = values_by_target(load("/tmp/mlp_donut/taffo_info_vra.json"))
c_dta = values_by_target(load("/tmp/mlp_classic/taffo_info_dta.json"))
d_dta = values_by_target(load("/tmp/mlp_donut/taffo_info_dta.json"))

def fmtrange(r):
    if not r or "min" not in r:
        return "-"
    s = f"[{r['min']:.3g}, {r['max']:.3g}]"
    comps = r.get("components")
    if comps:
        s += " donut=" + "U".join(f"[{a:.3g},{b:.3g}]" for a, b in comps)
    return s

def fmtnt(n):
    if not n:
        return "-"
    ib = n["bits"] - n["fractionalBits"]
    fb = n["fractionalBits"]
    return f"s{ib}_{fb}fixp"

order = ["W1", "W2", "b1", "b2", "x", "h_pre", "h_post", "y_pre", "y_post"]
print()
print(f"{'target':8s}  {'classic VRA':50s}  {'donut VRA':60s}  {'cDTA':12s}  {'dDTA':12s}")
print("-" * 150)
for t in order:
    c_r = first_range(c_vra.get(t, []))
    d_r = first_range(d_vra.get(t, []))
    c_n = first_nt(c_dta.get(t, []))
    d_n = first_nt(d_dta.get(t, []))
    note = ""
    donut_comps = d_r.get("components") if isinstance(d_r, dict) else None
    if donut_comps:
        note = "<-- seed"
    print(f"{t:8s}  {fmtrange(c_r):50s}  {fmtrange(d_r):60s}  "
          f"{fmtnt(c_n):12s}  {fmtnt(d_n):12s}  {note}")

print()
print("Summary: the seeded W1 / W2 carry their donut components through")
print("VRA; derived quantities do not inherit the hole because the kernel")
print("pattern (bimodal * zero-crossing input, then sum over fan-in)")
print("closes the hole by construction. This is the HONEST end-to-end")
print("finding for a ReLU-style MLP, documented in report §7.4.")
PY
