# RESUME — CMP50HX llama.cpp research

- Ночная кампания завершена: `campaign_body_complete` 2026-09-22T11:14:26+03:00, restore PASS 11:17:20.
- Продолжение (квалификация tensor): `cmp50-cont-campaign`, 2026-09-22 12:29:55 → 14:54:19 MSK.
- Отчёт продолжения: `summaries/CONTINUATION-REPORT.md`.
- Активный основной сервис: `llama-qwen.service`, health `:8081` = 200; профиль `profiles/current-selected.service` (НЕ менялся — tensor не продвигался).
- `llama-research.service`: inactive. `bonsai-2.service`: active, health `:8082` = 200.
- GPU: 4/4; GPU0–2 Gen2 x8, GPU3 Gen2 x4; Xid текущей загрузки 0.
- Итог по tensor: на `--parallel 1` быстрее layer по всем осям (p192k dec +87.7%, p250k dec +97.1%, soak +54.1%, conc1 +82.2%), но проигрывает масштабирование при 2+ клиентах (conc5 −33.5%).
- row-mode: ОПРОВЕРГНУТ — `device CUDA0 does not support split buffers` при загрузке модели.
- Незакрыто: CUDA 12.0 build matrix, Nsight, profiler-driven patches, clock sweep, elastic 256K.
- Вернуться к baseline: `sudo -n cp $R/rollback/llama-qwen.service.initial /etc/systemd/system/llama-qwen.service 2>/dev/null || sudo -n cp /home/fastchip/cmp50hx-llama-research/rollback/llama-qwen.service.initial /etc/systemd/system/llama-qwen.service && sudo -n systemctl daemon-reload && sudo -n systemctl restart llama-qwen.service`
