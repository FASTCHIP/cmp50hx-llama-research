#!/usr/bin/env python3
"""Pivot for the MMQ/cuBLAS threshold A/B: medians, spread, telemetry, same-run deltas.

Usage: python3 mmq_pivot.py RAW.jsonl [CONC.jsonl] [--control LABEL]
Prints only rows that passed the validity gate; the row count is printed first.
"""
import argparse
import collections
import json
import statistics as st
import sys

CONTROL = "mmq-ctl-ub512"


def load_rows(paths):
    rows, rejected = [], 0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue
            rows.append(r)
    return rows, rejected


def sm_clock(row, after=False):
    """gpu_before/after entries: idx, mem_MiB, pstate, power_W, sm_clk, mem_clk, util, temp, gen, width."""
    key = "gpu_after" if after else "gpu_before"
    vals = []
    for entry in row.get(key) or []:
        parts = [p.strip() for p in str(entry).split(",")]
        if len(parts) > 4 and parts[4].replace(".", "").isdigit():
            vals.append(float(parts[4]))
    return st.median(vals) if vals else None


def valid(row):
    if row.get("http_code") != 200:
        return False
    if row.get("prompt_tps") is None or row.get("decode_tps") is None:
        return False
    if not row.get("predicted_n"):
        return False
    return True


def norm_label(label, case):
    """Some harnesses bake the case into the label (`variant-p32k`); normalise it away."""
    suffix = "-" + case
    return label[: -len(suffix)] if label.endswith(suffix) else label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--conc", default=None)
    a = ap.parse_args()

    rows, rejected = load_rows(a.raw)
    for r in rows:
        r["label"] = norm_label(str(r.get("label", "")), str(r.get("case", "")))
    good = [r for r in rows if valid(r)]
    print(f"rows={len(rows)} valid={len(good)} rejected={len(rows) - len(good)}")

    groups = collections.defaultdict(list)
    for r in good:
        groups[(r["label"], r["case"])].append(r)

    print("\n=== по плечам и кейсам (медиана / min..max) ===")
    print(f"{'label':22s} {'case':6s} {'n':>2s} {'pp':>8s} {'pp_min..max':>17s} {'tg':>7s} {'tg_min..max':>15s} {'smMHz':>6s}")
    for key in sorted(groups):
        g = groups[key]
        pp = [x["prompt_tps"] for x in g]
        tg = [x["decode_tps"] for x in g]
        clock = [c for c in (sm_clock(x) for x in g) if c]
        print(f"{key[0]:22s} {key[1]:6s} {len(g):2d} {st.median(pp):8.2f} "
              f"{min(pp):7.1f}..{max(pp):7.1f} {st.median(tg):7.2f} {min(tg):6.1f}..{max(tg):6.1f} "
              f"{(st.median(clock) if clock else 0):6.0f}")

    print(f"\n=== дельты против контроля {a.control} (тот же прогон) ===")
    ctrl = {k[1]: v for k, v in groups.items() if k[0] == a.control}
    for key in sorted(groups):
        label, case = key
        if label == a.control or case not in ctrl:
            continue
        g, c = groups[key], ctrl[case]
        d_pp = (st.median(x["prompt_tps"] for x in g) / st.median(x["prompt_tps"] for x in c) - 1) * 100
        d_tg = (st.median(x["decode_tps"] for x in g) / st.median(x["decode_tps"] for x in c) - 1) * 100
        print(f"{label:22s} {case:6s} dpp={d_pp:+7.2f}%  dtg={d_tg:+7.2f}%  (n={len(g)} vs {len(c)})")

    if a.conc:
        crows, _ = load_rows([a.conc])
        cg = collections.defaultdict(list)
        for r in crows:
            if r.get("aggregate_tps"):
                cg[(r["label"], r["case"])].append(r["aggregate_tps"])
        print("\n=== конкурентность (aggregate tok/s) ===")
        cc = {k[1]: v for k, v in cg.items() if k[0] == a.control}
        for key in sorted(cg):
            v = cg[key]
            delta = ""
            if key[0] != a.control and key[1] in cc:
                delta = f"  d={(st.median(v)/st.median(cc[key[1]])-1)*100:+6.2f}%"
            print(f"{key[0]:22s} {key[1]:6s} n={len(v):2d} median={st.median(v):7.2f}{delta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
