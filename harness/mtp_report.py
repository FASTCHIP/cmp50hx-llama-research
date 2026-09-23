import json, glob, collections, statistics as st

g = collections.defaultdict(list)
for f in glob.glob('raw/mtp-ab-2*.jsonl'):
    if 'conc' in f:
        continue
    for l in open(f):
        r = json.loads(l)
        if r.get('http_code') == 200 and r.get('prompt_tps'):
            g[(r['label'], r['case'])].append(r)

base = {k[1]: st.median(x['decode_tps'] for x in v) for k, v in g.items() if k[0] == 'mtp-nodraft'}

print('%-11s %-6s %2s %7s %6s %6s %8s' % ('плечо', 'кейс', 'n', 'pp', 'tg', 'acc', 'dtg'))
for k in sorted(g):
    lab, case = k
    v = g[k]
    dn = sum(x.get('draft_n') or 0 for x in v)
    da = sum(x.get('draft_accepted') or 0 for x in v)
    pp = st.median(x['prompt_tps'] for x in v)
    tg = st.median(x['decode_tps'] for x in v)
    acc = (da / dn) if dn else 0.0
    d = ''
    if case in base:
        d = '%+.1f%%' % ((tg / base[case] - 1) * 100)
    print('%-11s %-6s %2d %7.1f %6.2f %6.3f %8s' % (lab, case, len(v), pp, tg, acc, d))
