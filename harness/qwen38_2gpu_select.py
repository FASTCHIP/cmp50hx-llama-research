#!/usr/bin/env python3
"""Select safe Pareto candidates from the two-GPU tuning campaign."""
from __future__ import annotations
import argparse, json, pathlib, statistics
from collections import defaultdict


def parse_used(snapshot: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for line in snapshot:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            out[int(parts[0])] = int(float(parts[1]))
    return out


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], expected: int = 3) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("label", "")].append(row)
    result: dict[str, dict] = {}
    for label, items in groups.items():
        good = [r for r in items if r.get("http_code") == 200 and r.get("prompt_tps") is not None and r.get("decode_tps") is not None]
        used = [parse_used(r.get("gpu_after") or []) for r in good]
        per_gpu_peak = {g: max((x.get(g, 0) for x in used), default=0) for g in (0, 1)}
        result[label] = {
            "rows": len(items),
            "good": len(good),
            "complete": len(items) == expected and len(good) == expected,
            "pp": statistics.median([r["prompt_tps"] for r in good]) if good else None,
            "decode": statistics.median([r["decode_tps"] for r in good]) if good else None,
            "gpu0_peak_mib": per_gpu_peak[0],
            "gpu1_peak_mib": per_gpu_peak[1],
            "max_used_mib": max(per_gpu_peak.values()),
            "imbalance_mib": abs(per_gpu_peak[0] - per_gpu_peak[1]),
        }
    return result


def select(summary: dict[str, dict], baseline: str = "split100", perf_floor: float = 0.98) -> tuple[str | None, dict[str, dict]]:
    base = summary.get(baseline)
    if not base or not base["complete"]:
        return None, summary
    for label, item in summary.items():
        item["eligible"] = bool(
            item["complete"]
            and item["pp"] >= base["pp"] * perf_floor
            and item["decode"] >= base["decode"] * perf_floor
        )
    eligible = [(label, item) for label, item in summary.items() if item.get("eligible")]
    if not eligible:
        return None, summary
    eligible.sort(key=lambda x: (x[1]["max_used_mib"], x[1]["imbalance_mib"], -x[1]["pp"], -x[1]["decode"]))
    return eligible[0][0], summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("--expected", type=int, default=3)
    ap.add_argument("--baseline", default="split100")
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    summary = summarize(load_rows(args.jsonl), args.expected)
    winner, summary = select(summary, args.baseline)
    doc = {"winner": winner, "baseline": args.baseline, "arms": summary}
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if winner else 2)


if __name__ == "__main__":
    main()
