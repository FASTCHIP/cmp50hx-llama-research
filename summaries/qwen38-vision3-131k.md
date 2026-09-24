# Qwen3.8-27B: текст+vision на GPU0–2, mmproj на GPU3

Стенд: ai100gb (192.168.50.9). Дата: 2026-09-24.
Сборка: llama.cpp-vbbb-mmq-thresh-6a4317e.

## Что принято

Юнит `qwen38-vision3-131k.service`
(active + enabled, порт :8081).

Ключевые аргументы:
```
CUDA_VISIBLE_DEVICES=0,1,2,3
--mmproj-device CUDA3 --mmproj-offload
--parallel 1 --ctx-size 131072
--cache-type-k q4_0 --cache-type-v q4_0
--flash-attn on
-b 2048 -ub 512
--split-mode layer --tensor-split 1,1,1,0
--spec-type none
```
Профиль сохранён:
`profiles/qwen38-vision3-131k.service`
sha256 aaf207158194f8d0…

## VRAM по картам (рабочий режим)

```
GPU0  6933 MiB / 20480
GPU1  7575 MiB / 20480
GPU2  8203 MiB / 20480
GPU3  1029 MiB / 10240
```
mmproj живёт на GPU3 — одна карта,
отдельная от весов и KV.

## Прежний клиентский путь (проверено)

Через Open WebUI `:4000/api/chat/completions`,
один пользовательский `sk-…`, model=Qwen3.8-27B:

```
text   HTTP 200  finish=stop  «Голубое»
image  HTTP 200  finish=stop  «Синий»
SSE    stream:true, 41 data, [DONE]
wrong key  HTTP 401
```
Оба запроса обслужил один PID (219173),
второй сервер не участвовал.

## Скорость (layer / без MTP)

13 строк, ответы HTTP 200, :8081:
```
case    decode tok/s  prefill tok/s
short        25.2        140.7
p4k          24.6        642.9
p32k         20.3        549.9
p64k         17.2        468.0
p120k        13.1        359.7
```
p120k — только 1 из 3 прогонов:
bench завис на «deferred tasks 1»
и был снят по таймауту (см. ниже).

32k-текст через WebUI: 54656 prompt
токенов, 117.9 s, HTTP 200.
Картинка сразу после — 5.9 s, «Синий».
Два клиента одновременно: оба 200,
суммарно 10.4 s (кэш прогрет).
`--parallel 1`: истинной параллели нет,
запросы обслуживаются по очереди.

## Инцидент

Прогон arm `layer-nomtp` встал на p120k
(«deferred tasks 1», prompt 0 tok/s,
gen 0 tok/s) и висел ~1 ч 50 мин.
Снят вручную; услуги возвращены.
KV q4_0 на 131072 держит слот после
длинного запроса: последовательные
near-limit заявки упираются в очередь.

## Откат

Старые юниты остановлены и **disabled**:
- llama-qwen.service
- bonsai-2.service

Бэкап эталона:
`backups/llama-qwen.service.20260924`
+ `llama-qwen.service.d.20260924`.
Возврат — отдельной командой.

## Решение по Bonsai

Bonsai на GPU3 выключен: вместе с
mmproj Qwen на 10 ГБ карте не
размещается. Порт :8082 не обслуживается,
пока старые юниты выключены.

## Xid

Новых Xid/AER с курсора: 0.
