#!/usr/bin/env python3
"""Full campaign aggregation: screening, long-context, concurrency, vision.

Reads every raw row of the campaign stamp, groups by label/case/class,
reports medians with n, deltas against the same-run reference arms, and
flags incomplete arms. Writes summaries/REPORT.md and summaries/final.json.
"""
import json, pathlib, statistics as st, sys
from collections import defaultdict

ROOT = pathlib.Path('/home/fastchip/bench/upstream-a02c7f5-speed')
STAMP = sys.argv[1] if len(sys.argv) > 1 else '20260924T2050Z'

REF_SHORT = 'base-a02c7f5'   # reference for short/p4k/p32k
REF_LONG  = 'base-long'      # reference for p64k/p120k/p192k
CASE_ORDER = ['short', 'p4k', 'p32k', 'p64k', 'p120k', 'p192k', 'p250k', 'conc4']
EXPECTED = {  # rows the arm was asked for
    'base-a02c7f5': 27, 'base2': 27, 'base3': 27, 'base-long': 9,
    'ub1024': 27, 'ub2048': 27, 'ub1024-rep5': 30, 'ub2048-rep5': 30,
    'thr0': 27, 'thr512': 27, 'cram4096': 27,
    'mtp1': 27, 'mtp5': 27, 'nospec': 27, 'mtp5-p4k': 9,
    'ub2048-long': 9, 'ub1024-long': 9, 'ub2048-p250k': 3,
    'tensor-mtp3': 27, 'tensor-long': 6,
    'ub2048-b4096': 27, 'ub2048-tb12': 27, 'ub2048-vision': 6,
}

def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None

def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

req = defaultdict(list)      # (label, case, cls) -> rows
conc = defaultdict(list)     # label -> batches
bad = defaultdict(list)

for p in sorted(ROOT.glob(f'raw/*-{STAMP}.jsonl')):
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        lab = r.get('label', p.stem)
        if 'requests' in r:                      # concurrency batch
            conc[lab].append(r)
            continue
        if r.get('http_code') != 200 or r.get('prompt_tps') is None or r.get('decode_tps') is None:
            bad[lab].append(r)
            continue
        req[(lab, r['case'], r['prompt_class'])].append(r)

labels = sorted({k[0] for k in req} | set(conc))
cases = [c for c in CASE_ORDER if any(k[1] == c for k in req)]

# per label/case medians
cell = defaultdict(dict)
for (lab, case, cls), rows in req.items():
    cell[lab][(case, cls)] = {
        'n': len(rows),
        'pp': med([r['prompt_tps'] for r in rows]),
        'dec': med([r['decode_tps'] for r in rows]),
        'acc': med([r.get('draft_acceptance') for r in rows]),
    }

def case_med(lab, case, field):
    vals = [v[field] for (c, cls), v in cell[lab].items() if c == case]
    return mean(vals)

def ref_val(ref, case, field):
    if ref not in cell:
        return None
    return case_med(ref, case, field)

out = {'stamp': STAMP, 'labels': {}, 'flags': {}, 'concurrency': {}, 'vision': {}}
lines = []
def emit(s=''):
    lines.append(s)

emit(f'# Сводка кампании upstream-a02c7f5-speed ({STAMP})')
emit()
emit('Все плечи: официальный upstream a02c7f5 + патч порога MMQ/cuBLAS, Qwen3.8-27B-UD-Q4_K_XL,')
emit('3 GPU (0,1,2), layer, tensor-split 1.2,1.2,0.6, parallel 4, общий KV 524288, per-slot 262144,')
emit('KV q8_0, FA on, MTP n=3 — если в названии плеча не сказано иное. 3 класса промптов,')
emit('медиана по репетам; сравнение только с эталоном, измеренным в том же прогоне.')
emit()

# ---- screening ----
screening = [c for c in cases if c in ('short', 'p4k', 'p32k')]
emit('## Скрининг: short / p4k / p32k (эталон base-a02c7f5)')
emit()
for case in screening:
    emit(f'### {case}')
    emit('```')
    emit('плечо        n   pp t/s   Δpp     dec t/s  Δdec')
    for lab in labels:
        if (case, 'prose') not in cell.get(lab, {}):
            continue
        n = min(v['n'] for (c, _), v in cell[lab].items() if c == case)
        pp, dec = case_med(lab, case, 'pp'), case_med(lab, case, 'dec')
        rp, rd = ref_val(REF_SHORT, case, 'pp'), ref_val(REF_SHORT, case, 'dec')
        dpp = 100 * (pp - rp) / rp if rp else None
        ddec = 100 * (dec - rd) / rd if rd else None
        emit(f'{lab:12s} {n:2d} {pp:8.1f} {dpp:+6.1f}% {dec:8.2f} {ddec:+6.1f}%')
        out['labels'].setdefault(lab, {}).setdefault(case, {}).update(
            {'n': n, 'pp': pp, 'decode': dec, 'pp_delta_pct': dpp, 'decode_delta_pct': ddec})
    emit('```')
    emit()

# ---- long context ----
longc = [c for c in cases if c in ('p64k', 'p120k', 'p192k', 'p250k')]
if longc:
    emit('## Длинный контекст (эталон base-long)')
    emit()
    emit('```')
    emit('плечо        case    n   pp t/s   Δpp     dec t/s  Δdec')
    for lab in labels:
        for case in longc:
            if (case, 'prose') not in cell.get(lab, {}):
                continue
            n = min(v['n'] for (c, _), v in cell[lab].items() if c == case)
            pp, dec = case_med(lab, case, 'pp'), case_med(lab, case, 'dec')
            rp, rd = ref_val(REF_LONG, case, 'pp'), ref_val(REF_LONG, case, 'dec')
            dpp = 100 * (pp - rp) / rp if rp else None
            ddec = 100 * (dec - rd) / rd if rd else None
            s1 = f'{dpp:+6.1f}%' if dpp is not None else '    —  '
            s2 = f'{ddec:+6.1f}%' if ddec is not None else '    —  '
            emit(f'{lab:12s} {case:6s} {n:2d} {pp:8.1f} {s1} {dec:8.2f} {s2}')
            out['labels'].setdefault(lab, {}).setdefault(case, {}).update(
                {'n': n, 'pp': pp, 'decode': dec, 'pp_delta_pct': dpp, 'decode_delta_pct': ddec})
    emit('```')
    emit()

# ---- concurrency ----
if conc:
    emit('## 4 одновременных клиента (разные промпты, 3 батча)')
    emit()
    emit('```')
    emit('плечо        адрегат t/s  ср. decode  n')
    for lab in labels:
        if lab not in conc:
            continue
        b = conc[lab]
        agg = med([x['aggregate_tps'] for x in b])
        per = med([q['decode_tps'] for x in b for q in x['requests']])
        emit(f'{lab:12s} {agg:11.2f} {per:11.2f} {len(b):2d}')
        out['concurrency'][lab] = {'n_batches': len(b), 'aggregate_tps_median': agg,
                                   'per_request_decode_median': per}
    emit('```')
    emit()

# ---- vision ----
vis = sorted((ROOT / 'baseline').glob('*vision*.json'))
if vis:
    emit('## Vision')
    emit()
    emit('```')
    for p in vis:
        try:
            d = json.loads(p.read_text())
            txt = str(d['choices'][0]['message'].get('content', '')).strip().replace('\n', ' ')[:40]
        except Exception as e:
            txt = f'parse_error {e}'
        emit(f'{p.name[:40]:42s} {txt}')
        out['vision'][p.name] = txt
    emit('```')
    emit()

# ---- data quality ----
emit('## Качество данных')
emit()
emit('```')
emit('плечо          строк  ожидалось  ошибок')
for lab in labels:
    if lab in conc:
        got = len(conc[lab]) * 0  # batches counted separately
    got = sum(len(rows) for (l, c, cl), rows in req.items() if l == lab)
    got_batches = len(conc.get(lab, []))
    exp = EXPECTED.get(lab)
    note = ''
    if exp is not None and got < exp:
        note = ' <-- НЕПОЛНОЕ'
    out['flags'][lab] = {'rows': got, 'batches': got_batches, 'expected': exp, 'bad': len(bad.get(lab, [])), 'incomplete': bool(note)}
    emit(f'{lab:14s} {got:5d} {str(exp):>9s} {len(bad.get(lab, [])):7d}{note}')
emit('```')
emit()
if bad:
    emit('Ошибочные строки:')
    emit('```')
    for lab, rows in sorted(bad.items()):
        for r in rows[:3]:
            emit(f'{lab:12s} {r.get("case")} {r.get("prompt_class")} rep{r.get("rep")} http={r.get("http_code")}')
    emit('```')

report = '\n'.join(lines)
(ROOT / 'summaries' / 'REPORT.md').write_text(report + '\n')
(ROOT / 'summaries' / 'final.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n')
print(report)
print()
print('wrote', ROOT / 'summaries' / 'REPORT.md')
