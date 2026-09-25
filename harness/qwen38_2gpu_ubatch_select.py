#!/usr/bin/env python3
"""Select the best non-baseline ubatch from complete screening rows."""
from __future__ import annotations
import argparse, json, pathlib, statistics
from collections import defaultdict

CASES = ("short", "p4k", "p32k")


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def summarize(rows: list[dict], expected_per_case: int = 6) -> dict[str, dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.get("label", "")][row.get("case", "")].append(row)
    out = {}
    for label, cases in grouped.items():
        doc = {"complete": True, "cases": {}}
        for case in CASES:
            items = cases.get(case, [])
            good = [r for r in items if r.get("http_code") == 200 and r.get("prompt_tps") is not None and r.get("decode_tps") is not None]
            cell = {
                "rows": len(items), "good": len(good),
                "pp": statistics.median([r["prompt_tps"] for r in good]) if good else None,
                "decode": statistics.median([r["decode_tps"] for r in good]) if good else None,
            }
            doc["cases"][case] = cell
            doc["complete"] &= len(items) == expected_per_case and len(good) == expected_per_case
        out[label] = doc
    return out


def select(summary: dict[str, dict], baseline: str = "ub512") -> tuple[str | None, dict[str, dict]]:
    base = summary.get(baseline)
    if not base or not base["complete"]:
        return None, summary
    candidates = []
    for label, arm in summary.items():
        if label == baseline:
            continue
        eligible = arm["complete"]
        for case in CASES:
            a, b = arm["cases"][case], base["cases"][case]
            eligible &= a["decode"] >= b["decode"] * 0.95
            eligible &= a["pp"] >= b["pp"] * 0.80
        arm["eligible"] = bool(eligible)
        if eligible:
            p32 = arm["cases"]["p32k"]["pp"] / base["cases"]["p32k"]["pp"]
            p4 = arm["cases"]["p4k"]["pp"] / base["cases"]["p4k"]["pp"]
            arm["score"] = 0.75 * p32 + 0.25 * p4
            candidates.append((label, arm))
    candidates.sort(key=lambda x: (-x[1]["score"], -x[1]["cases"]["p32k"]["decode"]))
    return (candidates[0][0] if candidates else None), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    summary = summarize(load(args.jsonl))
    winner, summary = select(summary)
    doc = {"alternate": winner, "baseline": "ub512", "arms": summary}
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if winner else 2)


if __name__ == "__main__":
    main()
