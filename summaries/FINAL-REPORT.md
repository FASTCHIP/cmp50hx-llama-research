# Итоговый отчёт CMP50HX llama.cpp

> Исторический отчёт ранней кампании. Пункты «не выполнено» и статус `tensor` здесь
> устарели после последующих A/B. Актуальные выводы: [FINAL-RESULTS-2026-09-23.md](FINAL-RESULTS-2026-09-23.md).

Сформирован: 2026-09-22T11:37:01+03:00 (MSK)

## Итог

Долговечный исполнитель `cmp50-llama-campaign-2.service` завершил тело кампании (`campaign_body_complete`) после 37 219 с (10:20:19), затем восстановил штатный сервис. Получено 294 JSONL-строки: 285 одиночных запросов и 9 строк конкурентного теста. Hard-stop событий, Xid и нарушений целостности P2P не было.

Безопасный выбранный профиль оставлен прежним: `/home/fastchip/cmp50hx-llama-research/profiles/current-selected.service`, установленная копия `/etc/systemd/system/llama-qwen.service`, rollback-снимок `/home/fastchip/cmp50hx-llama-research/rollback/llama-qwen.service.initial`.

Экспериментальный `tensor` дал большой выигрыш decode, но не получил обязательных проверок 192K/250K, vision, concurrency и soak, поэтому в штатный сервис не продвигался.

## Финальная приёмка

- `llama-qwen.service`: **active**, `http://127.0.0.1:8081/health` → **200**.
- На `0.0.0.0:8081` ровно один `llama-server`, PID 100930; `llama-research.service`: **inactive**.
- Аутентифицированный `/v1/models`: HTTP 200, модель `Qwen3.8-27B`, capabilities `completion,multimodal`.
- Финальный vision-тест: корректный ответ `Красный`, файл `/home/fastchip/cmp50hx-llama-research/baseline/vision-final.json`.
- `bonsai-2.service` был активен до кампании; восстановлен: **active**, `:8082/health` → **200**.
- 4/4 GPU видимы; GPU0–2 Gen2 x8, GPU3 Gen2 x4; Xid текущей загрузки: 0.

## Сводка по label

Медиана/минимум рассчитаны программно по всем строкам label. Для label со смешанными case prefill следует читать вместе с таблицей по case ниже.

| label | строк | decode median/min, tok/s | prefill median/min, tok/s |
|---|---:|---:|---:|
| prod-current | 30 | 24.58 / 17.71 | 500.30 / 80.69 |
| long-layer-q8 | 15 | 25.33 / 7.92 | 181.03 / 138.71 |
| topo-tensor-q8 | 9 | 38.69 / 33.02 | 97.26 / 81.00 |
| p2p-disabled | 9 | 25.35 / 20.89 | 166.17 / 133.96 |
| kv-q4q4 | 9 | 25.26 / 8.32 | 169.24 / 132.83 |
| batch4096 | 9 | 25.49 / 20.59 | 183.08 / 157.69 |
| ubatch1024 | 9 | 25.49 / 20.60 | 178.20 / 136.99 |
| prod-soak | 58 | 25.39 / 24.56 | 169.62 / 72.49 |
| stability-prod | 131 | 25.36 / 24.63 | 180.87 / 80.50 |
| final-prod | 3 | 24.91 / 24.69 | 91.05 / 81.68 |
| harness-smoke | 3 | 25.04 / 24.70 | 102.32 / 95.80 |

## Сопоставимые case

| Вариант / case | строки | decode median/min | prefill median/min | prompt_n median |
|---|---:|---:|---:|---:|
| prod-current / short | 9 | 25.16 / 25.01 | 97.19 / 80.69 | 48 |
| prod-current / p4k | 9 | 24.59 / 24.44 | 611.47 / 570.45 | 4 089 |
| prod-current / p32k | 9 | 20.81 / 20.74 | 500.45 / 499.93 | 32 769 |
| prod-current / p64k | 3 | 17.71 / 17.71 | 430.92 / 430.74 | 63 009 |
| long-layer-q8 / p192k | 3 | 9.75 / 9.73 | 269.86 / 269.85 | 196 609 |
| long-layer-q8 / p250k | 3 | 7.92 / 7.92 | 230.39 / 230.34 | 256 009 |
| kv-q4q4 / p250k | 3 | 8.32 / 8.32 | 231.06 / 231.02 | 256 009 |
| topo-tensor-q8 / short | 6 | 39.27 / 37.38 | 88.24 / 81.00 | 48 |
| topo-tensor-q8 / p32k | 3 | 33.22 / 33.02 | 508.18 / 507.77 | 32 769 |
| p2p-disabled / p32k | 3 | 21.00 / 20.89 | 507.09 / 507.03 | 32 769 |
| batch4096 / p32k | 3 | 20.88 / 20.59 | 508.06 / 506.73 | 32 769 |
| ubatch1024 / p32k | 3 | 20.85 / 20.60 | 517.36 / 516.51 | 32 769 |

Ключевые дельты против same-campaign baseline:

- `tensor` short decode: **+56.1%** (39.27 против 25.16 tok/s).
- `tensor` p32k decode: **+59.6%** (33.22 против 20.81 tok/s); p32k prefill: **+1.5%**.
- Q4/Q4 KV при ~256K: decode **+5.1%** (8.32 против 7.92 tok/s), prefill +0.3%.
- `-b 4096` не дал значимого p32k выигрыша; `-ub 1024` поднял p32k prefill примерно до 517.36 tok/s, но decode остался около baseline.
- P2P-disabled близок к layer baseline на p32k; это не доказывает ненужность P2P для `tensor`.

Конкурентный baseline, aggregate median/min: conc1 16.97/12.72, conc2 39.10/27.28, conc5 54.42/51.30 tok/s.

## Аппаратные измерения

GPU0/1/2 соответственно:

- D2D: 231.26 / 231.75 / 231.27 GiB/s.
- FP32: 12.38 / 12.15 / 12.89 TFLOP/s.
- FP16 tensor: 87.06 / 86.55 / 91.56 TFLOP/s.
- INT8 DP4A: 44.16 / 44.26 / 47.33 TOPS.

P2P для всех шести направлений GPU0–2: `can=1`, ноль mismatches. На 256 MiB peer bandwidth 2.4696 GiB/s, direct/staged ratio примерно 1.625–1.627.

## Бинарник и команды

- Fork/version: `verybigbadboy/llama.cpp`, commit `b7d4d85`, version `0.4.1-dev` build 5.
- `llama-server` SHA-256: `43e6eab36d1ac5b348f2a601705e2a39e35ce99cc277d341c34f3f6e1f65012b`.
- Активный unit и rollback имеют одинаковый SHA-256: `bcaf635b840b53d2d9fcd06c149564255e28e3a66f52a8e9e05a29178f4c41b3`.

Общая экспериментальная команда:

```text
/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/bin/llama-server --model /mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias Qwen3.8-27B --mmproj /mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf --mmproj-offload --host 0.0.0.0 --port 8081 --parallel 1 --ctx-size 262144 --temp 0 --flash-attn on --gpu-layers 99 --tensor-split 1,1,1 --spec-type none -t 6 -tb 6 --api-key-file /etc/llama-server.api-key
```

Дельты: baseline long `--split-mode layer --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512`; tensor меняет только `--split-mode tensor`; row меняет только `--split-mode row`; Q4/Q4 меняет только KV; batch4096 меняет только `-b`; ubatch1024 меняет только `-ub`; p2p-disabled добавляет только `NCCL_P2P_DISABLE=1`.

Полная штатная командная строка сохранена в `/home/fastchip/cmp50hx-llama-research/profiles/current-selected.service`.

## Raw-артефакты

- `/home/fastchip/cmp50hx-llama-research/raw/harness-smoke.jsonl` — 3
- `/home/fastchip/cmp50hx-llama-research/raw/baseline-20260921T215258Z.jsonl` — 30
- `/home/fastchip/cmp50hx-llama-research/raw/concurrency-20260921T215258Z.jsonl` — 9
- `/home/fastchip/cmp50hx-llama-research/raw/soak-20260921T215258Z.jsonl` — 58
- `/home/fastchip/cmp50hx-llama-research/raw/stability-20260921T215258Z.jsonl` — 131
- `/home/fastchip/cmp50hx-llama-research/raw/longctx-20260921T215258Z.jsonl` — 15
- `/home/fastchip/cmp50hx-llama-research/raw/topology-20260921T215258Z.jsonl` — 18
- `/home/fastchip/cmp50hx-llama-research/raw/elastic-20260921T215258Z.jsonl` — 27
- `/home/fastchip/cmp50hx-llama-research/raw/final-20260921T215258Z.jsonl` — 3
- `/home/fastchip/cmp50hx-llama-research/hardware/gpu-envelope.jsonl`
- `/home/fastchip/cmp50hx-llama-research/hardware/p2p-matrix.jsonl`
- `/home/fastchip/cmp50hx-llama-research/hardware/telemetry-soak-20260921T215258Z.csv`
- `/home/fastchip/cmp50hx-llama-research/logs/campaign.log`

## Незакрыто

Ночная кампания завершила минимальный безопасный runtime/hardware срез, но не весь исходный deep-optimization план. Не выполнены: CUDA 12.0/12.8 и NCCL/LTO build matrix на CT200; Nsight profiling; profiler-driven patches; clock sweep; полноценная elastic-policy проверка с 256K primary и secondary offload/queue; tensor long-context/vision/concurrency/soak; row-mode (дважды recoverable failure, raw-строк нет). До этих проверок `tensor` остаётся кандидатом, а не выбранным production-профилем.
