# CMP50HX llama.cpp research — режимы, скорости, сырые данные

Исследование производительности llama.cpp на трёх модифицированных CMP 50HX
(TU102, sm_75): prompt processing (prefill), decode, длинный контекст,
конкурентность и аппаратный конверт. Данные получены двумя прогонами на реальном
стенде; в репозитории — **все сырые строки** и сводные таблицы.

- Форк/версия: `verybigbadboy/llama.cpp`, commit `b7d4d85`, `0.4.1-dev` build 5
- Модель: `Qwen3.8-27B-UD-Q4_K_XL.gguf` + `mmproj-Qwen3.8-27B-Q8_0.gguf` (vision)
- Прогон 1 (ночная кампания): 2026-09-21 21:52 → 2026-09-22 11:17 UTC+3, ~10 ч 20 мин
- Прогон 2 (квалификация tensor): 2026-09-22 12:29 → 14:54 UTC+3
- Первые две кампании прошли без Xid; позднее FA tile64-испытание завершилось Xid 43. Полный набор raw на стенде включает последующие кампании; число строк и текущие выводы сверяйте по итоговой сводке, а не по этому историческому срезу.
- **Найденное ускорение prefill: [`summaries/MMQ-THRESHOLD-AB.md`](summaries/MMQ-THRESHOLD-AB.md)** — порог MMQ->cuBLAS на Turing: +8% prefill на 32K без потери decode, качества и коротких запросов.
- **Актуальный краткий отчёт: [`summaries/FINAL-RESULTS-2026-09-23.md`](summaries/FINAL-RESULTS-2026-09-23.md)** —
  сверенные числа, закрытые направления и ссылка на [план оптимизации](docs/superpowers/plans/2026-09-23-qwen38-sm75-throughput.md). Остальная часть README — исторический обзор; более ранние предложения по FA tile64 не применять.
- **Новая кампания (2026-09-24/25): официальный upstream `a02c7f5` + наш патч порога — [`summaries/UPSTREAM-A02C7F5-RESULTS-2026-09-25.md`](summaries/UPSTREAM-A02C7F5-RESULTS-2026-09-25.md)** —
  ubatch 2048: **+13.8% prefill на 32K** без потери decode; tensor-split: **+34% decode** ценой prefill; MTP n=3 подтверждён оптимальным (n=1 и n=5 хуже); thr0 подтверждает вклад патча (+8-10%).
  Сырые строки: [`raw/upstream-a02c7f5-20260924T2050Z/`](raw/upstream-a02c7f5-20260924T2050Z), логи плеч: [`logs/upstream-a02c7f5-20260924T2050Z/`](logs/upstream-a02c7f5-20260924T2050Z), манифест отпечатков: [`docs/upstream-a02c7f5-manifest.json`](docs/upstream-a02c7f5-manifest.json).
  Итоговая конфигурация стенда пока не менялась: активны `-ub 512`, MTP n=3, layer.

## Железо

```text
GPU0..2: CMP 50HX 20 GiB, sm_75, PCIe Gen2 x8, все пары через PHB (CPU), NVLink нет
GPU3   : CMP 50HX, PCIe Gen2 x4 (в этот single-model профиль не входит)
CPU    : Xeon E5-2620 v1, 6C/12T, 2.0-2.5 GHz
Драйвер: 610.43.03
```

Аппаратный конверт (GPU0/1/2):

```text
D2D        231.26 / 231.75 / 231.27 GiB/s
FP32       12.38 / 12.15 / 12.89 TFLOP/s
FP16 tc    87.06 / 86.55 / 91.56 TFLOP/s
INT8 DP4A  44.16 / 44.26 / 47.33 TOPS
P2P        6/6 направлений can=1, mismatches 0
peer BW    2.4696 GiB/s (256 MiB), direct/staged ≈ 1.625-1.627
```

## Профили

```text
production (llama-qwen.service)
  --parallel 5 --ctx-size 327680 --kv-unified --kv-unified-per-slot 65536
  --split-mode layer --tensor-split 1,1,1 --cache-type-k/v q8_0
  --flash-attn on --spec-type none -b 2048 -ub 512 --reasoning-effort low

research variants (все сравнения в таблицах ниже)
  --parallel 1 --ctx-size 262144 --split-mode <layer|tensor|row>
  --tensor-split 1,1,1 --cache-type-k/v q8_0 --spec-type none -b 2048 -ub 512
```

Отличие варианта от baseline — **ровно один флаг**, остальное идентично.

## Режимы split: короткий промпт и 32K

| режим | short decode | p32k decode | p32k prefill |
|---|---:|---:|---:|
| `layer` (baseline) | 25.16 | 20.81 | 500.45 |
| `tensor` | **39.27 (+56.1%)** | **33.22 (+59.6%)** | 508.18 (+1.5%) |
| `row` | — | — | **не поддерживается** |

`row` отвергнут на загрузке модели: `device CUDA0 does not support split buffers`.

## Длинный контекст: `tensor` против `layer`

| case | режим | prefill | vs layer | decode | vs layer |
|---|---|---:|---:|---:|---:|
| p192k | layer | 269.86 | — | 9.75 | — |
| p192k | tensor | **341.1** | **+26.4%** | **18.30** | **+87.7%** |
| p250k | layer | 230.39 | — | 7.92 | — |
| p250k | tensor | **304.0** | **+32.0%** | **15.61** | **+97.1%** |

По 3 класса промпта (prose/code/architecture), все ответы HTTP 200, разброс
между классами ничтожный (prefill 341.0–341.4 и 304.0–304.2).

## Конкурентность — определяющий компромисс

| клиентов | `layer` | `tensor` | дельта |
|---|---:|---:|---:|
| 1 | 16.97 | 30.91 | **+82.2%** |
| 2 | 39.10 | 35.83 | −8.4% |
| 5 | **54.42** | 36.18 | **−33.5%** |

`tensor` насыщается (35.8 → 36.2 между 2 и 5), `layer` продолжает расти
(17 → 39 → 54).

## KV-кэш, батчи, P2P (одно изменение против layer baseline)

| вариант | p32k decode | p32k prefill |
|---|---:|---:|
| layer q8_0 (baseline) | 20.81 | 500.45 |
| `-b 4096` | 20.88 | 508.06 |
| `-ub 1024` | 20.85 | **517.36** |
| `NCCL_P2P_DISABLE=1` | 21.00 | 507.09 |
| q4_0 KV на 250K | 8.32 (+5.1% к 7.92) | 231.06 |

## Устойчивость (soak)

| режим | строк | decode median | примечание |
|---|---:|---:|---|
| prod-soak (`layer`) | 58 | 25.39 | baseline |
| tensor soak | 135 | **39.14 (+54.1%)** | 0 не-200 ответов |

## Вывод

- **`tensor` быстрее в однослотовом режиме**: одиночный длинный запрос (+87…+97%
  decode на 192K/250K, +26…+32% prefill), устойчивый короткий режим (+54%),
  один клиент (+82%).
- **`layer` быстрее при нескольких слотах**: `--parallel 5` в проде — это ровно
  тот режим, где `tensor` теряет 33.5% (насыщается на ~36 tok/s против 54).
- Продовый профиль поэтому **не менялся**: остался `--split-mode layer`.
  Продвижение `tensor` требует отдельного теста на реальном числе слотов (5).
- `row` неприменим архитектурно. `-b 4096`, `-ub 1024` и отключение P2P в пределах
  шума; `-ub 1024` даёт небольшой прирост prefill (~3%), но не decode.
- **Фиксация режима (23.09.2026).** Кампания `cont9` подтвердила `tensor` на прод-бинарнике
  (decode 32.92 против 20.76 tok/s, +58.5%, при равном префилле) — но она же напомнила
  про потолок конкурентности. Поэтому закреплено два профиля: `layer` как рабочая форма
  5-слотового сервиса (`llama-qwen.service`, не менялся) и `tensor` как профиль для
  одиночного/длинноконтекстного доступа (`systemd/llama-qwen-tensor.service`, выключен
  по умолчанию). Подробности, гейт качества и оставшиеся рычаги —
  [`summaries/FINAL-RESULTS-2026-09-23.md`](summaries/FINAL-RESULTS-2026-09-23.md).

## Структура репозитория

```text
data/raw/*.jsonl        444 сырых строки (по одной на запрос): prompt_n, prompt_tps,
                        decode_tps, decode_ms, prompt_ms, wall_s, http_code, gpu_before/after
data/hardware/          gpu-envelope.jsonl, p2p-matrix.jsonl, telemetry soak CSV
summaries/              FINAL-REPORT.md (ночная кампания), CONTINUATION-REPORT.md (квалификация tensor)
docs/RESUME.md          состояние исследования и что осталось незакрытым
harness/                bench_request.py, bench_concurrency.py, vision_smoke.py,
                        run_variant.sh (preguard + restore trap), campaign.sh
hardware/               gpu_envelope.cu, p2p_matrix.cu (исходники микробенчей)
profile/                current-selected.service — действующий продовый unit
```

### Схема сырой строки

```json
{"schema":1,"label":"tensor-q8","case":"p192k","prompt_class":"prose","rep":1,
 "prompt_n":196606,"prompt_tps":341.4,"decode_tps":17.87,"http_code":200,
 "wall_s":585.4,"boot_id":"...","content_sha256":"...","gpu_before":["..."],"gpu_after":["..."]}
```

## Сборки: CUDA 12.0 против 12.8

Матрица собрана на CT 200 (`builds/manifest-cuda120.jsonl`, `manifest-cuda128.jsonl`):
по 4 варианта — `nccl-on/off` × `lto` × `fa-all`. Все PASS, в обеих версиях
`endbr64=5` (столько же в рабочих сборках; регрессии по ISA нет).

Замер в одинаковом конфиге (`nccl-on-lto-off`, layer, q8_0, `-b 2048 -ub 512`):

| метрика | cuda128 | cuda120 | дельта |
|---|---:|---:|---:|
| p32k decode | 20.90 | 20.97 | +0.3% |
| p32k prefill | 507.8 | 507.4 | −0.1% |
| short decode, разблокировано | 25.26 | 25.34 | +0.3% |
| short decode @1700 МГц | 25.06 | 25.15 | +0.4% |
| short decode @2100 МГц | 25.56 | 25.64 | +0.3% |

Разницы нет — всё в пределах шума. Остаёмся на CUDA 12.8.

## Clock sweep: не рычаг

| частоты ядра | decode медиана | к разблокированному |
|---|---:|---:|
| разблокировано | 25.26 | — |
| залочено 1700 МГц | 25.06 | −0.8% |
| залочено 2100 МГц | 25.56 | +1.1% |

Эффект ≤1%: decode упирается в память, а не в частоту ядра. Лок работает
технически (`nvidia-smi -i N -lgc`, проверено 1500 и 2100 МГц, `-rgc` возвращает
буст), но ускорения не даёт.

## Elastic 256K: ОПРОВЕРГНУТО

Действующий профиль (`--parallel 5 --kv-unified-per-slot 65536`) отвергает
одиночный запрос больше 65536 токенов:

| prompt tokens | ответ |
|---|---|
| 4 087 | HTTP 200 |
| 65 527 | HTTP 200 |
| 70 007 | **HTTP 400** `exceed_context_size_error` |
| 196 607 | **HTTP 400** `exceed_context_size_error` |

Эластичности нет: запрос не встаёт в очередь и не вытесняет secondary — он
отвергается до обработки. «256K» здесь — суммарная ёмкость пяти слотов, а не
доступный одному запросу контекст.

## Nsight: профилирование запрещено вендором

```text
==ERROR== ERR_NVCMPGPU - Profiling is not supported on the NVIDIA
Crypto Mining Processors (CMP) of the target device 0.
```

`nsys` не падает, но не собирает данные вообще: в отчёте нет ни одной
CUPTI-таблицы — проверено и на `llama-server`, и на минимальном CUDA-микробенче.
Инструментальное профилирование ядер на этом стенде недоступно, поэтому
«profiler-driven patches» в исходном виде невозможны: нет данных профилировщика,
на которые можно опираться.

## Незакрытое (после этого этапа)

CUDA 12.0 matrix — закрыта. Clock sweep — закрыт. Elastic 256K — закрыт
(опровергнут). Nsight — закрыт как недоступный.
Остаётся: продвижение `tensor` в прод (решение владельца), гипотетические
патчи без профилировщика.

---

## Strata (2026-07 →): Qwen3.8-Flash-Next на тех же трёх картах

Помимо llama.cpp-кампаний, на этом же стенде поднят движок **Strata** (Niko1221/Strata) с полной
Qwen3.8-Flash-Next **Q2_0** (125B MoE, 6B активных): 100 % экспертов в VRAM трёх 20-GB карт,
контекст **256K**, **2 параллельных запроса**.

- Установка, параметры, правила запуска обеих моделей (Coder IQ1_M и полная Q2_0), патч движка
  для параллели и бенчмарки decode/prefill: **[docs/strata-ai100gb-setup.md](docs/strata-ai100gb-setup.md)**
- Патч `parallel >= 2` + 100 % резидентность экспертов (upstream #776/#792/#845):
  **[patches/strata-batch-zerodoorbell-792.patch](patches/strata-batch-zerodoorbell-792.patch)**
- Юниты и конфиги: `systemd/strata-*.service`, `systemd/rgminer-gpu3-only.conf`, `docs/strata-*.json`

Коротко по скорости (3x20 GB, майнер выключен, ~1900 МГц): decode **80-82 tok/s** (короткий промпт),
**62 tok/s** на 100K-контексте, префилл **2.1-2.4K tok/s** на 30-100K промптах, чекпоинты
переиспользуют 100K истории за 0.6 с.
