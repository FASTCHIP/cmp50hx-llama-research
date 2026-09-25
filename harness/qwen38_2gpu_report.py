#!/usr/bin/env python3
"""Final report for the two-GPU balance campaign (stamp 20260925T064249Z)."""
import json, pathlib, statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / 'raw' / 'qwen38-2gpu-balance-20260925T064249Z'
OUT = ROOT / 'summaries' / 'QWEN38-2GPU-BALANCE-20260925.md'


def rows(name):
    p = RAW / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def gpu_peak(r, idx):
    for line in r.get('gpu_after') or []:
        parts = [x.strip() for x in line.split(',')]
        if int(parts[0]) == idx:
            return int(parts[1])
    return None


L = []
def w(s=''):
    L.append(s)


w('# Qwen3.8-27B на двух CMP 50HX: баланс слоёв, ubatch и потолок контекста')
w()
w('Стенд ai100gb (192.168.50.9), кампания `20260925T064249Z`. Бинарь — официальный')
w('upstream `a02c7f5` + патч порога MMQ/cuBLAS (`GGML_CUDA_TURING_CUBLAS_MIN_M=256`).')
w('Всё измерено в одном окне, сравнение только с эталоном того же прогона.')
w()
w('## Итоговый профиль (установлен)')
w()
w('```')
w('CUDA_VISIBLE_DEVICES=0,1')
w('--parallel 2 --ctx-size 262144 --kv-unified')
w('--kv-unified-per-slot 262144')
w('--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on')
w('--split-mode layer --tensor-split 1.10,0.90')
w('--spec-type draft-mtp --spec-draft-n-max 3')
w('-t 6 -tb 6 -b 2048 -ub 512')
w('--cache-ram 4096          <-- исправлено (было 16384)')
w('```')
w()

# ---- split screen ----
w('## 1. Распределение слоёв (p32k, три класса, ubatch 512)')
w()
w('```')
w('split       GPU0   GPU1   перекос   PP t/s   dec t/s')
for lab, split in (('split100', '1.00/1.00'), ('split110', '1.10/0.90'),
                   ('split115', '1.15/0.85'), ('split120', '1.20/0.80')):
    rs = [r for r in rows('split-screen.jsonl') if r['label'] == lab and r.get('http_code') == 200]
    if not rs:
        continue
    p0 = max(gpu_peak(r, 0) or 0 for r in rs)
    p1 = max(gpu_peak(r, 1) or 0 for r in rs)
    w(f'{split}   {p0:5d}  {p1:5d}   {abs(p0-p1):5d}   {med([r["prompt_tps"] for r in rs]):7.1f}   {med([r["decode_tps"] for r in rs]):7.2f}')
w('```')
w()
w('Перекос снижен с 3172 до 134 МиB при потере PP 0.1% и decode 0.1%.')
w()

# ---- split long ----
w('## 2. Длинный контекст (p120k)')
w()
w('```')
w('split       PP t/s   dec t/s')
for lab, split in (('split100-long', '1.00/1.00'), ('splitbest-long', '1.10/0.90')):
    rs = [r for r in rows('split-long.jsonl') if r['label'] == lab and r.get('http_code') == 200]
    if rs:
        w(f'{split}   {med([r["prompt_tps"] for r in rs]):7.1f}   {med([r["decode_tps"] for r in rs]):7.2f}')
w('```')
w()

# ---- ubatch ----
w('## 3. Ubatch при фиксированном split 1.10/0.90')
w()
w('```')
w('ubatch   short PP   p4k PP   p32k PP   dec t/s')
for lab, ub in (('ub256', 256), ('ub320', 320), ('ub384', 384), ('ub512', 512)):
    cells = {}
    for case in ('short', 'p4k', 'p32k'):
        rs = [r for r in rows('ubatch-screen.jsonl')
              if r['label'] == lab and r['case'] == case and r.get('http_code') == 200]
        cells[case] = med([r['prompt_tps'] for r in rs])
    rs = [r for r in rows('ubatch-screen.jsonl') if r['label'] == lab and r.get('http_code') == 200]
    w(f'{ub:5d}   {cells["short"]:8.1f}   {cells["p4k"]:6.1f}   {cells["p32k"]:7.1f}   {med([r["decode_tps"] for r in rs]):7.2f}')
w('```')
w()
w('Длинный контекст (p120k), ub512 против ub384:')
w()
w('```')
w('плечо        PP t/s   dec t/s')
for lab in ('ub512-long', 'ubalt-long'):
    rs = [r for r in rows('ubatch-long.jsonl') if r['label'] == lab and r.get('http_code') == 200]
    if rs:
        w(f'{lab:12s} {med([r["prompt_tps"] for r in rs]):7.1f}   {med([r["decode_tps"] for r in rs]):7.2f}')
w('```')
w()
w('Уменьшение ubatch не дало прироста PP на этой сборке: на p120k варианты равны,')
w('на коротких и средних промптах 512 быстрее на 4-8%. Выбран 512.')
w()

# ---- mode ----
w('## 4. Layer (pipeline) против tensor')
w()
w('```')
w('режим    p120K PP   dec t/s   2 клиента t/s')
for lab, name in (('mode-layer', 'layer'), ('mode-tensor', 'tensor')):
    rs = [r for r in rows('mode-finalists.jsonl') if r['label'] == lab and r.get('http_code') == 200]
    rg = [r for r in rs if r['case'] == 'p120k']
    cc = [r['aggregate_tps'] for r in rows('mode-finalists-conc.jsonl') if r['label'] == lab]
    w(f'{name:7s}  {med([r["prompt_tps"] for r in rg]):8.1f}   {med([r["decode_tps"] for r in rg]):7.2f}   {med(cc):9.2f}')
w('```')
w()
w('Tensor даёт decode +40% и суммарную скорость двух клиентов +14%, но теряет 10.5% PP.')
w('При приоритете «один 256K плюс нормальный второй клиент» выбран layer: просадка PP')
w('бьёт по обработке длинных промптов, а выигрыш decode достижим и без смены режима.')
w()

# ---- p250 ----
w('## 5. Проверка на 256K (256006 токенов, 128 выходных)')
w()
w('```')
w('профиль            PP t/s   dec t/s   VRAM GPU0/GPU1')
allrows = rows('p250-final.jsonl') + rows('p250-disambig.jsonl')
for lab, name in (('p250-ref', '1.00/1.00 ub512'),
                  ('p250-split512', '1.10/0.90 ub512'),
                  ('p250-win', '1.10/0.90 ub384'),
                  ('ramcap-p250', '1.10/0.90 ub512 ramcap')):
    rs = [r for r in allrows if r['label'] == lab and r.get('http_code') == 200]
    if not rs:
        continue
    r = rs[0]
    w(f'{name:22s} {r["prompt_tps"]:7.1f}   {r["decode_tps"]:7.2f}   {gpu_peak(r,0)}/{gpu_peak(r,1)}')
w('```')
w()
w('При фиксированном ubatch 512 распределение 1.10/0.90 не меняет скорость на 256K')
w('(258.26 против 258.44 PP), но убирает перекос VRAM. С `ubatch 384` получается +2.2%')
w('на 256K — меньше порога 3%, установленного в плане, поэтому отклонено.')
w()

# ---- ramcap ----
w('## 6. Дефект хост-RAM и его устранение')
w()
w('Симптом: сразу после 256K-запроса, при запуске второго клиента,')
w('система убивала сервер по OOM.')
w()
w('```')
w('OOM: Killed process llama-server')
w('anon-rss: 11875524 kB (11.9 GB)')
w('host RAM total: 15909 MB')
w('--cache-ram 16384 (16 GiB) + --cache-idle-slots')
w('```')
w()
w('Механика: при появлении нового запроса свободный слот с 256K-контекстом')
w('сохраняется в prompt-кэш в оперативной памяти (состояние ~8.9 GB при q8_0),')
w('и второй клиент выводит потребление за пределы 15.9 GB.')
w()
w('После ограничения `--cache-ram 4096`:')
w()
w('```')
w('256K запрос      257.8 PP   15.90 dec   HTTP 200')
w('два клиента      41.6 / 42.1 / 44.7 ток/с суммарно')
w('per-request dec  23.0-25.5 ток/с')
w('пик RSS          3.63 GB (было 11.9 GB)')
w('свободно RAM     ~10.3 GB, OOM 0, Xid 0')
w('```')
w()
w('`--cache-ram` — это потолок кэша, а не зарезервированная память; по прежним')
w('замерам он не влияет на скорость, поэтому снижение бесплатно.')
w()

# ---- gates ----
w('## 7. Приёмка через реальный клиентский путь')
w()
w('```')
w('текст    HTTP 200  finish=stop  «Синий»')
w('vision   HTTP 200  finish=stop  синяя -> «Синий»')
w('vision   HTTP 200  finish=stop  красная -> «Красный»')
w('SSE      HTTP 200  33 data-строки, завершение [DONE]')
w('ключ     HTTP 401 на неверном ключе')
w('серверов 1 процесс на :8081')
w('Xid/OOM  0 после курсора')
w('```')
w()
w('## 8. Закрытые вопросы')
w()
w('```')
w('- ubatch < 512 не повышает PP на этой сборке')
w('- tensor split проигрывает по PP на production-глубине')
w('- cache-ram 16384 несовместим с 15.9 GB хоста при работе с 256K')
w('```')
w()
w('## 9. Открытые вопросы')
w()
w('```')
w('- два параллельных 256K-диалога не проверялись:')
w('  пул 262144 допускает один почти-256K или ~два 128K')
w('- поведение при трёх и более клиентах не измерялось')
w('- вклад split и ubatch в 256K проверен, но с одним повтором на плечо')
w('```')

OUT.write_text('\n'.join(L) + '\n')
print('\n'.join(L))
print('\nwrote', OUT)
