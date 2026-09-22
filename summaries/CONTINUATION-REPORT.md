# Отчёт-продолжение: квалификация `tensor` — 2026-09-22

Продолжение ночной кампании. Закрывает ворота, которые блокировали продвижение
`tensor` в штатный профиль, и один ранее необъяснённый отказ.

- Исполнитель: `cmp50-cont-campaign.service`, старт 2026-09-22T12:29:55+03:00,
  конец 14:54:19+03:00.
- Harness: тот же `run_variant.sh` (preguard + restore trap); rollback не понадобился.
- Общее отличие варианта от baseline: только `--split-mode tensor` вместо `layer`.
  Тот же бинарник `llama.cpp-vbbb-b7d4d85-sm75`, `-b 2048 -ub 512`,
  `--cache-type-k/v q8_0`, `--spec-type none`, ctx 262144, `--parallel 1`.
- Восстановление: `llama-qwen` health 200, `bonsai-2` health 200, 4/4 GPU,
  Xid за буст = 0.

## Gate 1 — long context: PASS

| case | prompt_n | prefill | vs layer | decode | vs layer |
|---|---:|---:|---:|---:|---:|
| p192k (3 класса) | 196 606–196 609 | 341.1 | **+26.4%** | 18.30 | **+87.7%** |
| p250k (3 класса) | 256 006–256 009 | 304.0 | **+32.0%** | 15.61 | **+97.1%** |

Baseline `long-layer-q8`: p192k 269.86 / 9.75; p250k 230.39 / 7.92.
Все 6 строк HTTP 200. Разброс между классами промпта ничтожный
(prefill 341.0–341.4 и 304.0–304.2; decode 17.87–18.32 и 15.60–15.62),
т.е. это не выброс.

## Gate 2 — vision + concurrency: PASS (с оговоркой)

Vision: корректный ответ `Красный`.

| concurrency | tensor median | layer baseline | delta |
|---|---:|---:|---:|
| 1 | 30.91 | 16.97 | **+82.2%** |
| 2 | 35.83 | 39.10 | −8.4% |
| 5 | 36.18 | 54.42 | **−33.5%** |

`tensor` насыщается: 35.83 → 36.18 между 2 и 5 клиентами, роста нет.
`layer` продолжает масштабироваться: 16.97 → 39.10 → 54.42.

## Gate 3 — soak: PASS

45 последовательных short-запросов × 3 класса = 135 строк, не-200 ответов **0**.

- decode median 39.14 tok/s (min 36.92) против `prod-soak` 25.39 → **+54.1%**

## row-mode: ОПРОВЕРГНУТ (найдена первопричина)

Ранее завершался дважды без raw-строк и без объяснения. Причина получена:

```text
E llama_model_load: error loading model: device CUDA0 does not support split buffers
E llama_model_load_from_file_impl: failed to load model
E cmn common_init_: failed to load model
E srv  llama_server: exiting due to model loading error
```

`--split-mode row` не поддерживается CUDA-бэкендом этой сборки на этих картах:
падение на загрузке модели через ~13 с, до поднятия listener. Это архитектурное
ограничение, а не флак и не нехватка памяти. Вариант закрыт как неприменимый.

## Вывод по `tensor`

- Побеждает: одиночный длинный запрос (decode +87…+97% на 192K/250K, prefill +26…+32%),
  устойчивый короткий режим (+54% decode), один клиент (+82%).
- Проигрывает: масштабирование при 2+ одновременных клиентах (−8% при 2, −33% при 5).

Штатный профиль работает с `--parallel 5` (пять слотов), а варианты кампании
запускались с `--parallel 1`, т.е. в однослотовом режиме. В однослотовом
режиме `tensor` лучше по всем измеренным осям, но продовый режим —
многослотовый, и именно там `tensor` теряет (conc5 −33.5%). Поэтому `tensor`
не продвигается: нужен отдельный замер на реальном числе слотов (5).

**Поправка к промежуточному выводу этой кампании.** В первом отчёте было
сказано, что штатный сервис работает с `--parallel 1`. Это неверно: реальный
`llama-qwen.service` использует `--parallel 5 --ctx-size 327680
--kv-unified-per-slot 65536`. Все сравнения ниже сделаны на `--parallel 1`,
что и является главным ограничением этих измерений.

## Что осталось незакрытым

CUDA 12.0 build matrix, Nsight profiling, profiler-driven patches, clock sweep,
полная elastic-проверка 256K primary + secondary offload/queue.

## Артефакты

- `raw/tensor-qual-20260922T092955Z.jsonl` — 6
- `raw/tensor-conc-20260922T092955Z.jsonl` — 9
- `raw/tensor-soak-20260922T092955Z.jsonl` — 135
- `baseline/vision-tensor-20260922T092955Z.json`
- `logs/cont.log`, `logs/campaign.log`, `logs/trace-tensor-q8-*.log`
