"""Verify the macro forward pass against the reference dumps.

Runs the traced forward pass inside the workbook for a fixed prompt and
compares every intermediate tensor against reference/. Two stages:

  1. each building block on its own, driven through the Probe module
  2. the whole stack, all 156 intermediates of one forward pass

Deviations are reported in computation order, and the summary names the
EARLIEST tensor that exceeds the tolerance. That is the only one worth
looking at: everything downstream of a wrong tensor is wrong as a
consequence, so chasing the largest deviation usually means debugging a
symptom several layers away from the cause.

Exit code 1 on any deviation, so this can gate a change.
"""

import argparse
import math
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WB = os.path.join(BASE_DIR, "build", "excelgpt.xlsm")
DEFAULT_REF = os.path.join(BASE_DIR, "reference")
DEFAULT_PROMPT = "To be, or not to be"
TRACE_SHEET = "97_Trace"
PROBE_SHEET = "98_Probe"
PROBE_OUT_ROW = 700          # results go well below any input block
MASK_LIMIT = 1e29            # values beyond this are the causal-mask filler


def load_manifest(ref_dir: str) -> dict:
    # ref_dir: the reference directory. Returns: {name: (rows, cols)} in the
    # order the dump wrote them, which is computation order.
    manifest = {}
    with open(os.path.join(ref_dir, "manifest.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            manifest[parts[0]] = (int(parts[1]), int(parts[2]))
    return manifest


def load_ref(ref_dir: str, name: str, shape: tuple) -> np.ndarray:
    return np.genfromtxt(os.path.join(ref_dir, name + ".csv"),
                         delimiter=",").reshape(shape)


def deviation(got: np.ndarray, want: np.ndarray) -> float:
    # Masked positions hold -1e30 as a stand-in for negative infinity; they
    # carry no information and are left out of the comparison.
    sel = np.abs(want) < MASK_LIMIT
    if not sel.any():
        return 0.0
    return float(np.abs(got[sel] - want[sel]).max())


def check_primitives(app, book, ref_dir: str, manifest: dict, tol: float) -> list:
    # Drives each building block on its own through the Probe module.
    # Returns: a list of (label, deviation, ok).
    sheet = book.sheets[PROBE_SHEET]
    sheet.clear()
    run = app.api.Run
    out = []

    def put(values, row):
        a = np.atleast_2d(np.asarray(values, dtype=np.float64))
        rng = sheet.range((row, 1), (row + a.shape[0] - 1, a.shape[1]))
        rng.value = a.tolist()
        return rng.address

    def out_addr(rows, cols):
        return sheet.range((PROBE_OUT_ROW, 1),
                           (PROBE_OUT_ROW + rows - 1, cols)).address

    def take(addr):
        return np.array(sheet.range(addr).options(ndim=2).value, dtype=np.float64)

    def ref(name):
        return load_ref(ref_dir, name, manifest[name])

    def record(label, got, want):
        dev = deviation(np.asarray(got, dtype=np.float64),
                        np.asarray(want, dtype=np.float64))
        out.append((label, dev, dev < tol))

    t = manifest["03_x_input"][0]
    width = manifest["03_x_input"][1]
    hidden = manifest["L0_31_fc"][1]
    head_dim = manifest["L0_11_q_h0"][1]

    # LayerNorm
    addr = put(ref("03_x_input"), 1)
    o = out_addr(t, width)
    run("Probe.ProbeLayerNorm", addr, "L0_LN1_W", "L0_LN1_B", o)
    record("LayerNorm", take(o), ref("L0_10_ln1"))

    # Linear with bias, checked against every head of Q
    addr = put(ref("L0_10_ln1"), 1)
    o = out_addr(t, 3 * width)
    run("Probe.ProbeMatMulAddBias", addr, "L0_ATTN_W", "L0_ATTN_B", o)
    qkv = take(o)
    for h in range(width // head_dim):
        record(f"Linear+bias, Q head {h}",
               qkv[:, h * head_dim:(h + 1) * head_dim], ref(f"L0_11_q_h{h}"))
    record("Linear+bias, K head 0",
           qkv[:, width:width + head_dim], ref("L0_12_k_h0"))
    record("Linear+bias, V head 0",
           qkv[:, 2 * width:2 * width + head_dim], ref("L0_13_v_h0"))

    # Causal mask
    addr = put(ref("L0_14_scores_scaled_h0"), 1)
    o = out_addr(t, t)
    run("Probe.ProbeCausalMask", addr, o)
    masked = take(o)
    record("Causal mask", masked, ref("L0_15_scores_masked_h0"))
    upper = masked[np.triu_indices(t, k=1)]
    out.append(("Causal mask: upper triangle is the filler", 0.0,
                bool(np.all(upper <= -MASK_LIMIT))))

    # Softmax
    addr = put(ref("L0_15_scores_masked_h0"), 1)
    o = out_addr(t, t)
    run("Probe.ProbeSoftmaxRows", addr, o)
    probs = take(o)
    record("Softmax", probs, ref("L0_16_attn_probs_h0"))
    out.append(("Softmax: rows sum to one",
                float(np.abs(probs.sum(axis=1) - 1.0).max()),
                bool(np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12)))
    out.append(("Softmax: upper triangle is zero",
                float(np.abs(probs[np.triu_indices(t, k=1)]).max()),
                bool(np.all(probs[np.triu_indices(t, k=1)] == 0.0))))

    # GELU
    addr = put(ref("L0_31_fc"), 1)
    o = out_addr(t, hidden)
    run("Probe.ProbeGelu", addr, o)
    record("GELU", take(o), ref("L0_32_gelu"))

    # Plain matrix product, via the attention scores
    q = ref("L0_11_q_h0")
    k = ref("L0_12_k_h0")
    a_addr = put(q, 1)
    b_addr = put(k.T, t + 5)
    o = out_addr(t, t)
    run("Probe.ProbeMatMul", a_addr, b_addr, o)
    record("Matrix product", take(o) / math.sqrt(head_dim),
           ref("L0_14_scores_scaled_h0"))

    # Scalar edge cases the array checks cannot reach
    for z in (-40.0, -1.5, 0.0, 0.7, 25.0):
        record(f"tanh({z:g})", [[run("Probe.ProbeTanh", z)]], [[math.tanh(z)]])

    sheet.clear()
    return out


def check_full_stack(app, book, ref_dir: str, manifest: dict,
                     prompt: str, tol: float) -> tuple:
    # Runs the traced forward pass and compares every intermediate.
    # Returns: (rows, missing, unexpected) where rows is
    # (name, deviation, ok) in computation order.
    import re

    label_re = re.compile(r"^(\S+)\s+\((\d+) x (\d+)\)$")
    sheet = book.sheets[TRACE_SHEET]
    app.api.Run("Gpt.RunTrace", prompt, TRACE_SHEET)

    last = sheet.used_range.last_cell.row
    column = sheet.range((1, 1), (last, 1)).options(ndim=2).value

    rows = []
    seen = []
    unexpected = []
    for i, (value,) in enumerate(column, start=1):
        if not isinstance(value, str):
            continue
        m = label_re.match(value.strip())
        if not m:
            continue
        name, nr, nc = m.group(1), int(m.group(2)), int(m.group(3))
        if name not in manifest:
            unexpected.append(name)
            continue
        seen.append(name)
        if manifest[name] != (nr, nc):
            rows.append((name, float("nan"), False))
            continue
        got = np.array(sheet.range((i + 1, 1), (i + nr, nc)).options(ndim=2).value,
                       dtype=np.float64)
        dev = deviation(got, load_ref(ref_dir, name, (nr, nc)))
        rows.append((name, dev, dev < tol))

    missing = [n for n in manifest if n not in set(seen)]
    return rows, missing, unexpected


def main() -> None:
    p = argparse.ArgumentParser(description="Verify the macro forward pass")
    p.add_argument("--workbook", type=str, default=DEFAULT_WB)
    p.add_argument("--reference", type=str, default=DEFAULT_REF)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--verbose", action="store_true",
                   help="list every tensor, not just the worst ones")
    args = p.parse_args()

    import xlwings as xw

    if not os.path.exists(args.workbook):
        print(f"{args.workbook} is missing -- build and inject first",
              file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(args.reference)

    app = xw.App(visible=False, add_book=False)
    try:
        app.display_alerts = False
        book = app.books.open(args.workbook)

        print(f"Prompt: {args.prompt!r}   tolerance: {args.tol:.0e}")
        print()
        print("Stage 1 -- building blocks on their own")
        primitives = check_primitives(app, book, args.reference, manifest, args.tol)
        for label, dev, ok in primitives:
            if args.verbose or not ok:
                print(f"  {label:<40} {dev:11.3e}  {'ok' if ok else 'DEVIATION'}")
        p_bad = [r for r in primitives if not r[2]]
        p_worst = max((r[1] for r in primitives if not math.isnan(r[1])), default=0.0)
        print(f"  {len(primitives)} checks, largest deviation {p_worst:.3e}, "
              f"{len(p_bad)} failed")
        print()

        print("Stage 2 -- the whole stack, in computation order")
        rows, missing, unexpected = check_full_stack(
            app, book, args.reference, manifest, args.prompt, args.tol)
        for name, dev, ok in rows:
            if args.verbose or not ok:
                shown = "  shape" if math.isnan(dev) else f"{dev:11.3e}"
                print(f"  {name:<32} {shown}  {'ok' if ok else 'DEVIATION'}")

        book.save()
        book.close()
    finally:
        app.quit()

    bad = [r for r in rows if not r[2]]
    devs = [r[1] for r in rows if not math.isnan(r[1])]
    worst = max(devs, default=0.0)
    worst_name = next((n for n, d, _ in rows if d == worst), "-")

    print(f"  {len(rows)} tensors compared, largest deviation {worst:.3e} "
          f"({worst_name})")
    if missing:
        print(f"  NOT TRACED: {len(missing)} {missing[:5]}")
    if unexpected:
        print(f"  NOT IN THE REFERENCE: {len(unexpected)} {unexpected[:5]}")
    print()

    failed = bool(p_bad or bad or missing or unexpected)
    if not failed:
        print(f"PASS -- {len(primitives)} block checks and {len(rows)} tensors "
              f"within {args.tol:.0e}")
        sys.exit(0)

    # The earliest deviation is the one to work on; the rest follow from it.
    if bad:
        first_name, first_dev, _ = bad[0]
        position = [r[0] for r in rows].index(first_name) + 1
        print(f"FAIL -- earliest deviating tensor: {first_name} "
              f"(position {position} of {len(rows)}, deviation {first_dev:.3e})")
        print("        Start there. Everything after it is downstream of this.")
    else:
        print("FAIL -- see the deviations above")
    sys.exit(1)


if __name__ == "__main__":
    main()
