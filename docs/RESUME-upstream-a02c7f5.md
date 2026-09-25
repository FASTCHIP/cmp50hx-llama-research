# RESUME — кампания upstream-a02c7f5-speed

Обновлено: 2026-09-25 (МСК, ночь). План: /home/fastchip/.hermes/plans/2026-09-24_201120-cmp50hx-prefill-decode.md

## Что измеряем

Официальная сборка a02c7f5 + патч порога MMQ/cuBLAS (GGML_CUDA_TURING_CUBLAS_MIN_M=256).
Стенд 192.168.50.9, прод не затрагивается. Фиксировано: Qwen3.8-27B-UD-Q4_K_XL, mmproj,
3 GPU (0,1,2), layer, tensor-split 1.2,1.2,0.6, parallel 4, общий KV 524288, per-slot 262144,
KV q8_0, FA on, --cache-ram 16384 (кроме плеча cram4096), MTP n=3 (кроме плеч спекуляции).

## Etap 1 (ЗАКРЫТ): screening, 27 строк на плечо, 3 репета

Метрика: prompt_tps / decode_tps, медиана по 3 классам.

```
Плечо      pp ср.  dec ср.
base       399.5   36.44
base2      398.3   36.45   контроль дрейфа
base3      398.2   36.41   контроль после всех плеч
thr512     398.4   35.99   контроль границы M
cram4096   398.1   36.40   нейтрально
ub1024     436.5   35.43   +9.3% pp
ub2048     449.2   36.55   +12.4% pp
mtp1       400.7   33.14   хуже
mtp5       419.0   32.85   хуже
nospec     444.7   23.79   MTP n=3 = +53% decode к nospec
thr0       367.2   36.50   патч выключен: -8.1%
```

Лестница prefill на 32k: thr0 456.4 -> base 495.8 -> ub1024 541.0 -> ub2048 564.3.
На 4k: 520.0 -> 575.4 -> 639.7 -> 663.5.

Итог: ubatch — рабочий рычаг prefill на этой сборке (прежний вывод о null относился к чистому
MMQ и к threshold-патчу не переносится). MTP n=3 — оптимум. Аномалия: у mtp5 на p4k/проза
prefill 739 (+28%), остальные классы не изменились — требует отдельной проверки.

## Etap 2 (В РАБОТЕ): длинные контексты, tensor, concurrency, vision

Плечи: ub2048-long, ub1024-long, base-conc, ub2048-conc, tensor-mtp3, tensor-long,
ub2048-vision, ub1024-conc, ub2048-b4096, ub2048-tb12.
Проверено по исходникам: upstream не запрещает tensor split при квантованном KV,
требует только flash_attn (включён). Архитектура qwen проходит llm_arch_supports_sm_tensor.

## Инструменты

- Рабочий каталог: /home/fastchip/bench/upstream-a02c7f5-speed (manifest.json с отпечатками)
- Runner: harness/run_variant.sh (копия проверенного, ROOT переtarгетен), цепочки chain_screen.sh и chain_phase2.sh
- Сводка: harness/aggregate_screen.py STAMP [--write]
- Запуск: sudo -n systemctl stop bench-chain / bench-phase2 (units: Type=oneshot, durable)
- Плечи идут через systemd-run llama-research.service на :8081; сервис llama-qwen.service
  восстанавливается trap-ом после каждого плеча, Xid считается от cursor

## Откат к эталону

sudo -n cp /home/fastchip/bench/upstream-a02c7f5-speed/state/llama-qwen.service.<arm>.snapshot /etc/systemd/system/llama-qwen.service
sudo -n systemctl daemon-reload && sudo -n systemctl restart llama-qwen.service
(первичный откат к форковой сборке: .../llama-qwen.service.bak-20260924-upstream)
