#!/usr/bin/env bash
# Smoke test for the `scalar(range_union(...))` annotation syntax.
# Runs test/donut_ranges/donut_annotated.c through the full TAFFO driver
# twice (flag on + flag off), then checks three things:
#   1. The seeded annotation produces the right `components` in the
#      VRA JSON.
#   2. With -vra-donut-ranges, handleMul's self-square fast path
#      collapses w*w to the tight single interval [0.04, 0.64].
#   3. With -vra-donut-ranges, handleMul on two independent donuts
#      preserves the [-0.64,-0.04] U [0.04, 0.64] hole through the
#      multiplication.
# With the flag off the same annotation still parses but degrades to a
# classic hull; the script asserts the hull is sound.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAFFO="$ROOT/install/bin/taffo"
SRC="$ROOT/test/donut_ranges/donut_annotated.c"
TMP_ON="/tmp/donut_annot"
TMP_OFF="/tmp/donut_annot_off"

if [[ ! -x "$TAFFO" ]]; then
  echo "taffo driver not found at $TAFFO; build and install first." >&2
  exit 1
fi

rm -rf "$TMP_ON" "$TMP_OFF"
mkdir -p "$TMP_ON" "$TMP_OFF"

echo "---- DONUT ON  : $TAFFO -Xvra -vra-donut-ranges ..."
"$TAFFO" -O2 -c -emit-llvm -Xvra -vra-donut-ranges \
  -temp-dir "$TMP_ON" -o "$TMP_ON/donut_annotated.ll" "$SRC" \
  >"$TMP_ON/taffo.stdout" 2>"$TMP_ON/taffo.stderr"

echo "---- DONUT OFF : $TAFFO ..."
"$TAFFO" -O2 -c -emit-llvm \
  -temp-dir "$TMP_OFF" -o "$TMP_OFF/donut_annotated.ll" "$SRC" \
  >"$TMP_OFF/taffo.stdout" 2>"$TMP_OFF/taffo.stderr"

python3 - <<'PY'
import json, sys

def load(path):
    with open(path) as f:
        return json.load(f)

def values_with_target(doc):
    out = {}
    for k, v in doc.get("values", {}).items():
        info = v.get("info") if isinstance(v, dict) else None
        if not isinstance(info, dict):
            continue
        tgt = info.get("target")
        rng = info.get("range") or {}
        if tgt:
            out.setdefault(tgt, []).append((k, rng))
    return out

def check(cond, msg):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)

on = values_with_target(load("/tmp/donut_annot/taffo_info_vra.json"))
off = values_with_target(load("/tmp/donut_annot_off/taffo_info_vra.json"))

EXPECTED_COMPONENTS = [[-0.8, -0.2], [0.2, 0.8]]

print("-- seed annotations survive into VRA (flag ON) --")
for name in ("bimodal_w", "bimodal_w2"):
    hits = on.get(name, [])
    seeds = [r for _, r in hits if r.get("components") == EXPECTED_COMPONENTS]
    check(len(seeds) >= 1,
          f"{name}: at least one VRA value carries "
          f"components={EXPECTED_COMPONENTS}")

print("-- arithmetic propagation (flag ON) --")
w2 = [r for _, r in on.get("w_squared", []) if r.get("min") is not None]
tight_sq = [r for r in w2
            if abs(r["min"] - 0.04) < 1e-6 and abs(r["max"] - 0.64) < 1e-6
            and "components" not in r]
check(len(tight_sq) >= 1,
      "handleMul self-square: w*w collapses to [0.04, 0.64] "
      "(single classic interval)")

wp = [r for _, r in on.get("w_product", []) if r.get("components")]
def near(a, b, eps=1e-6): return abs(a - b) < eps
def matches_product(comps):
    if len(comps) != 2: return False
    c1, c2 = comps
    return (near(c1[0], -0.64) and near(c1[1], -0.04)
            and near(c2[0],  0.04) and near(c2[1],  0.64))
donut_prod = [r for r in wp if matches_product(r["components"])]
check(len(donut_prod) >= 1,
      "handleMul two-donut product preserves the "
      "[-0.64,-0.04] U [0.04, 0.64] hole")

print("-- seed carried even with flag OFF (serialisation) --")
for name in ("bimodal_w", "bimodal_w2"):
    hits = off.get(name, [])
    seeds = [r for _, r in hits if r.get("components") == EXPECTED_COMPONENTS]
    check(len(seeds) >= 1,
          f"{name}: seed VRA value still has components in flag-off JSON")

print("-- arithmetic falls back to classic (flag OFF) --")
w2_off = [r for _, r in off.get("w_squared", []) if r.get("min") is not None]
classic_sq = [r for r in w2_off
              if "components" not in r
              and abs(r["min"] - 0.0) < 1e-6
              and abs(r["max"] - 0.64) < 1e-6]
check(len(classic_sq) >= 1,
      "flag OFF: w*w uses classic hull [0, 0.64]")

wp_off = [r for _, r in off.get("w_product", []) if r.get("min") is not None]
classic_prod = [r for r in wp_off
                if "components" not in r
                and abs(r["min"] + 0.64) < 1e-6
                and abs(r["max"] - 0.64) < 1e-6]
check(len(classic_prod) >= 1,
      "flag OFF: w*w_other uses classic hull [-0.64, 0.64] "
      "(no hole preservation)")

print("\nAll range_union annotation assertions passed.")
PY
