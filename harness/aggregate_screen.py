#!/usr/bin/env python3
"""Aggregate campaign raw rows into a narrow delta table.

Usage: aggregate_screen.py STAMP [--write]
Groups by (label, case, prompt_class), medians of prompt_tps / decode_tps /
draft_acceptance, and reports paired deltas against the base label.
"""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path('/home/fastchip/bench/upstream-a02c7f5-speed')
BASE_LABEL = 'base-a02c7f5'

def load(stamp):
    rows = []
    for p in sorted(ROOT.glob(f'raw/*-{stamp}.jsonl')):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('http_code') != 200 or r.get('prompt_tps') is None:
                continue
            rows.append(r)
    return rows

def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else float('nan')

def main():
    stamp = sys.argv[1]
    rows = load(stamp)
    groups = {}
    for r in rows:
        key = (r['label'], r['case'], r['prompt_class'])
        groups.setdefault(key, []).append(r)

    labels = sorted({k[0] for k in groups})
    cases = sorted({k[1] for k in groups}, key=lambda c: ['short', 'p4k', 'p32k', 'p64k', 'p120k', 'p192k', 'p250k'].index(c) if c in ['short', 'p4k', 'p32k', 'p64k', 'p120k', 'p192k', 'p250k'] else 99)

    base = {}
    for (lab, case, cls), rs in groups.items():
        if lab == BASE_LABEL:
            base[(case, cls)] = (med([r['prompt_tps'] for r in rs]), med([r['decode_tps'] for r in rs]))

    out = {'stamp': stamp, 'rows': len(rows), 'groups': {}, 'deltas': {}}
    print(f'rows={len(rows)} labels={len(labels)}')
    for lab in labels:
        print()
        print(f'== {lab}')
        print('case  class        n  pp_tps   dec_tps  pp%    dec%')
        for case in cases:
            for cls in ('prose', 'code', 'architecture'):
                rs = groups.get((lab, case, cls))
                if not rs:
                    continue
                pp, dec = med([r['prompt_tps'] for r in rs]), med([r['decode_tps'] for r in rs])
                acc = med([r.get('draft_acceptance') for r in rs])
                b = base.get((case, cls))
                dpp = (100 * (pp - b[0]) / b[0]) if b and b[0] else float('nan')
                ddec = (100 * (dec - b[1]) / b[1]) if b and b[1] else float('nan')
                print(f'{case:5s} {cls:12s} {len(rs):2d} {pp:8.1f} {dec:8.2f} {dpp:+6.1f} {ddec:+6.1f}')
                out['groups'].setdefault(lab, {})[f'{case}/{cls}'] = {
                    'n': len(rs), 'pp_tps': pp, 'decode_tps': dec, 'acceptance': acc,
                    'pp_delta_pct': dpp, 'decode_delta_pct': ddec}
    # aggregate per label (mean of case/class medians, equal weight)
    for lab in labels:
        pp = [v['pp_tps'] for v in out['groups'][lab].values()]
        dec = [v['decode_tps'] for v in out['groups'][lab].values()]
        out['deltas'][lab] = {'pp_tps_mean': sum(pp) / len(pp) if pp else None,
                              'decode_tps_mean': sum(dec) / len(dec) if dec else None}
    print()
    print('== summary (mean of case medians)')
    print('label            pp_tps   dec_tps')
    for lab in labels:
        d = out['deltas'][lab]
        print(f'{lab:16s} {d["pp_tps_mean"]:8.1f} {d["decode_tps_mean"]:8.2f}')

    if '--write' in sys.argv:
        (ROOT / 'summaries' / f'screen-{stamp}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
        print('\nwrote', ROOT / 'summaries' / f'screen-{stamp}.json')

if __name__ == '__main__':
    main()
