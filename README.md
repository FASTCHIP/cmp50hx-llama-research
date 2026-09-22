# CMP50HX llama.cpp research — режимы, скорости, сырые данные

Исследование производительности llama.cpp на трёх модифицированных CMP 50HX
(TU102, sm_75): prompt processing (prefill), decode, длинный контекст,
конкурентность и аппаратный конверт. Данные получены двумя прогонами на реальном
стенде; в репозитории — **все сырые строки** и сводные таблицы.

- Форк/версия: `verybigbadboy/llama.cpp`, commit `b7d4d85`, `0.4.1-dev` build 5
- Модель: `Qwen3.8-27B-UD-Q4_K_XL.gguf` + `mmproj-Qwen3.8-27B-Q8_0.gguf` (vision)
- Прогон 1 (ночная кампания): 2026-09-21 21:52 → 2026-09-22 11:17 UTC+3, ~10 ч 20 мин
- Прогон 2 (квалификация tensor): 2026-09-22 12:29 → 14:54 UTC+3
- Всего: **444 сырых строки**, hard-stop событий 0, Xid 0, нарушений целостности P2P 0

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

## Незакрытое

CUDA 12.0 build matrix, Nsight profiling, profiler-driven patches, clock sweep,
полная elastic-проверка 256K (primary + secondary offload/queue).
