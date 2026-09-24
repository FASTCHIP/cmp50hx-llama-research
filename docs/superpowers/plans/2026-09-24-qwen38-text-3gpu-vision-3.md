# Qwen3.8 text+vision, GPU0–2 + GPU3 — план

> **Для исполнения:** выполнять по задачам; это план, не утверждение, что схема уже работает. Не запускать ещё один runner поверх текущего `/home/fastchip/nv_vision3_ab.sh` — на момент актуализации он ещё работал; сначала дождаться `restore_done` и проверить восстановление обоих исходных сервисов.

**Цель:** тот же агент Qwen3.8-27B понимает и текст, и изображения; основная GGUF-модель работает на GPU0,1,2, её совместимый vision-проектор — на GPU3. Один поток (`--parallel 1`), общий контекст `--ctx-size 131072`, KV K/V `q4_0`; подобрать лучший воспроизводимый decode. Клиентам сохранить прежнюю точку входа Open WebUI, пользовательский API-ключ и ID Qwen.

**Архитектура:** ОДИН `llama-server` для Qwen text+vision. Проектор — часть этого же сервера: `--mmproj ... --mmproj-offload --mmproj-device CUDA3`; это не отдельная модель и не второй HTTP-процесс. Основная модель/её KV/MTP остаются на GPU0–2; GPU3 с 10240 МиБ хранит mmproj и связанные vision-буферы. Копия 27B-модели только на GPU3 не поместится (~16.35 ГиБ весов). `bonsai-2` НЕ является vision-бэкендом Qwen и никогда не должен получать картинки клиента, выбравшего Qwen. Он может остаться самостоятельной моделью только при доказанной совместимости по памяти; иначе отключается на время выбранного Qwen-профиля с зафиксированным изменением доступных моделей.

**Клиентский контракт:** существующий Open WebUI на хосте `:4000`, API `/api/chat/completions`, принимает персональные WebUI `sk-...`. Его приватный провайдер Qwen указывает на `:8081/v1` и использует ВНУТРЕННИЙ ключ llama.cpp. Один и тот же клиентский URL, ключ и `model=Qwen3.8-27B` должны обслуживать как text, так и image запросы — они не делятся между двумя Qwen-процессами, а попадают в один multimodal `llama-server`. Второй сервис (если существует) выбирается только явно по своему model ID, не подменяет Qwen незаметно. Open WebUI уже поддерживает два backend URL; отдельный маршрутизатор **не нужен** для этой задачи. Если требуется автоматическое распределение запросов ОДНОЙ модели между несколькими llama-процессами, это отдельная задача с физическим ресурсным гейтом: второй экземпляр Qwen на GPU3 невозможен на этих 10 ГБ. Нельзя обещать такое распределение в текущем внедрении.

**Хост и артефакты:** стенд `192.168.50.9`, GPU0–2 по 20480 МиБ, GPU3 10240 МиБ/Gen2 x4. GGUF `/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`; совместимый проектор `/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf`; проверенный бинарник `/home/fastchip/llama.cpp-vbbb-mmq-thresh-6a4317e/bin/llama-server`. Бенч через `harness/run_variant.sh` и `harness/bench_request.py` в `/home/fastchip/cmp50hx-llama-research`, не через `llama-bench`. Предыдущее задание по чистке бэкапов выполнено: единственный актуальный снимок установленного юнита и drop-in — `backups/llama-qwen.service.20260924` и `.d.20260924`; перед любым изменением сверять с текущим установленным юнитом.

## Карта файлов

- Этот файл: `docs/superpowers/plans/2026-09-24-qwen38-text-3gpu-vision-3.md` — план, чекпойнты, ссылки на результаты.
- `harness/run_variant.sh`, `harness/bench_request.py` — существующие восстановление/генерация JSONL; менять только при доказанном дефекте.
- `/home/fastchip/cmp50hx-llama-research/profiles/qwen38-vision3-131k.service` — проект юнита, не устанавливать до acceptance.
- `/home/fastchip/cmp50hx-llama-research/raw/qwen38-vision3-*.jsonl` — сырые строки экспериментов, append-only.
- `/home/fastchip/cmp50hx-llama-research/summaries/qwen38-vision3-131k.md` — узкая итоговая сравнительная таблица, placement и end-to-end тесты.
- `/home/fastchip/cmp50hx-llama-research/logs/campaign.log` и журнал `llama-research.service` — фазы восстановления и устройство назначения, без ключей/изображений в тексте отчёта.

---

### Задача 1. Зафиксировать исходное состояние и клиентский контракт

**Результат:** известно, что именно будет сохранено при смене конфигурации.

- [x] Дождаться завершения уже запущенного `nv_vision3_ab.sh` и записи `restore_done`; не запускать параллельный runner. Снять `systemctl cat/show -p ExecStart -p Environment -p ActiveState llama-qwen.service bonsai-2.service`, `/health`, `/v1/models`, `nvidia-smi` memory/links, `/proc/<pid>/cmdline`, boot ID, cursor kernel journal. Удостовериться, что активный `bonsai-2` вернулся в прежнее состояние; при неудачном restore сначала восстановить именно его и Qwen, затем продолжать.
- [x] Сверить sha256 текущего unit/drop-in с единственным бэкапом. В зафиксированном на 24.09 эталоне `llama-qwen.service` уже использовал `--parallel 4 --ctx-size 524288`, KV q8, layer 1.2/1.2/0.6, MTP n3 и mmproj. Это исторический снимок: заново прочитать живой юнит, не переносить те флаги как факт текущего состояния.
- [x] В режиме read-only определить **реальный клиентский путь** (публичный URL, если он есть, и LAN `:4000/api`), исходный model ID Qwen и возможности image в Open WebUI; две существующие пары внутренних provider URL `:8081/v1` / `:8082/v1` и соответствующих backend-ключей прочитать БЕЗ вывода самих секретов. Проверить один настоящий текстовый и один image-запрос Qwen через WebUI с прежним пользовательским ключом, зафиксировать статусы и ответ; использовать именно `/api/chat/completions` для пользователя, не backend `/v1/chat/completions`. Ключи не писать в файлы исследования или логи.
- **Выполнено 24.09:** runner завершился (exit 0); в `logs/campaign.log` — `variant=layer restore_done` 12:52:56 и `variant=tensor restore_done` 13:13:22, оба `health=200 llama_count=2 new_xid=0`. После restore `llama-qwen` и `bonsai-2` active, `/health` на `:8081`/`:8082` = 200, новых Xid нет, VRAM на эталоне (~18.7/18.9/16.6 ГиБ GPU0–2, 8891 МиБ GPU3). Снимки в `snapshots/vision3-task1/` (cat/show юнитов без секретов, boot_id, journal_cursor).
- **sha256:** живой `llama-qwen.service` == `backups/llama-qwen.service.20260924` (MATCH); файловых бэкапов для `bonsai-2.service` и drop-in `gen2-order.conf` нет (no-backup) — зафиксировано как есть, каталог `llama-qwen.service.d.20260924` на месте.
- **Клиентский контракт:** пользовательский ключ берётся из таблицы `api_key` в `webui.db` (значение нигде не печаталось и не сохранялось); model ID `Qwen3.8-27B`; провайдеры `host.docker.internal:8081/v1` и `:8082/v1` (backend-ключи прочитаны без вывода). Через `:4000/api/chat/completions`: текст HTTP 200 finish=stop «Голубое.»; image (data-URL PNG, зелёный 128×128) HTTP 200 finish=stop «Зелёный» — ответил Qwen (эхо model=Qwen3.8-27B), не Bonsai; неверный ключ → 401. Лог: `snapshots/vision3-task1/webui_contract.txt`.
- **Предварительный A/B этого runner (не финальный baseline):** tensor быстрее layer во всех 15 парах: медиана tg ≈45.6 против ≈34.6 (диапазоны 41.2–60.0 и 27.8–45.8), draft_acc до ~0.79. Финальное сравнение — по протоколу задачи 4; сырые строки `raw/nv-layer.jsonl`, `raw/nv-tensor.jsonl`.

### Задача 2. Предпроверка размещения проектора

**Результат:** валидный флаг `-mmdev` не выдаётся за проверенное размещение без CUDA- и image-замера.

- [x] На том же бинарнике прочитать `--help`, `--list-devices`, `--version` со stderr, startup logs. Проверить `--mmproj-device CUDA3`, `--split-mode layer|tensor`, `--tensor-split`, q4_0 K/V, MTP; preflight с `/nonexistent.gguf` (синтаксис валиден, ошибка лишь о файле).
- [x] Начать с `CUDA_VISIBLE_DEVICES=0,1,2,3 --tensor-split 1,1,1,0 --mmproj-device CUDA3 --mmproj-offload` при `--gpu-layers 99`, `--parallel 1 --ctx-size 131072`, K/V q4_0, `--flash-attn on`, `-b 2048 -ub 512`, `--spec-type none`. **Не** считать `--device CUDA0,CUDA1,CUDA2` доказательством изоляции: на этом форке исключение GPU3 из буферных типов может сломать `-mmdev` или перенос MTP. Сравнить только после загрузки, где размещены веса/KV/mmproj по журналу и VRAM каждой карты.
- [x] Для каждого запуска отдельный writable каталог prompt-cache, не существующий рабочий `kvcache`; проверить `NCCL`, PCIe Gen2 x8 у GPU0–2/x4 у GPU3 и отсутствие новых Xid/AER после kernel-journal cursor. Если модель или проектор оказались не там — variant FAIL; корректировать параметры по одному и повторять полный тест.
- **Выполнено 24.09 (placement v3t2-place):** preflight layer/tensor на `/nonexistent.gguf` — синтаксис валиден (ошибка только о файле), флаг `--mmproj-device` подтверждён в `--help` форка. Guarded-запуск через `run_variant.sh` (layer, ts 1,1,1,0, mmproj→CUDA3, q4_0 KV, 131k, spec none, свежий `kvcache-v3t2-place`): bench_PASS 14:18:45, restore_done health=200 llama_count=2, новых Xid/AER = 0 с cursor. VRAM под нагрузкой: GPU0 6725 / GPU1 7367 / GPU2 7995 / GPU3 987 МиБ — mmproj на GPU3, веса/KV на GPU0–2 (ниже эталона: parallel 1/131k против parallel 4/524k). serve-check: health=200, `/v1/models` → Qwen3.8-27B capabilities completion+multimodal, `/completion` отвечает. GPU3 линк Gen2 x4 (5.0 GT/s, width 4). Артефакты: `snapshots/vision3-task2/` (vram-place.txt, serve-check.txt, journal-place-*.txt, preflight-*.txt, help.txt).

### Задача 3. Функциональный multimodal-gate

**Результат:** *Qwen*, а не Bonsai, реально понимает картинку и текст через один PID.

- [x] Гонять варианты через единственный `harness/run_variant.sh` с `trap ... EXIT`: снимок состояния обоих сервисов; stop/VRAM-free poll; запуск `llama-research.service` на :8081; ждать `/health=200`; image+text и текст; stop; восстановление исходного unit и каждого изначально активного sibling, повторный health и реальный ответ. Это стенд; всё равно не оставлять исходный сервис или `bonsai-2` остановленными. Не пускать второй runner пока первый в restore.
- [x] Создать воспроизводимую локальную картинку с очевидным предметом/цветом. Отправить Qwen `POST /v1/chat/completions` (правильный backend key) с `image_url` data URL, 800+ max_tokens либо `chat_template_kwargs.enable_thinking=false`, проверить ответ `finish_reason=stop`, ненулевой `content` и конкретное правильное описание. Повторить с другой картинкой и контрольным текстом без картинки; убедиться, что оба обслужены одним Qwen PID, а GPU3 заняла проекция, не другой сервер. `/health=200` и `loaded multimodal model` сами по себе недостаточны.
- [x] При OOM зафиксировать карту и размер буфера из журнала; изменить доли модели 0–2 (сохраняя нулевую долю GPU3) либо `-ub`, каждый фактор отдельно, и заново проверить image и decode. Установить peak VRAM GPU3 при обработке изображения; запуск `bonsai-2` одновременно НЕ обещать до отдельного guarded-теста headroom/параллельной нагрузки.
- **Выполнено 24.09 (gate v3t2-gate):** воспроизводимые fixtures `fix_red.png`/`fix_blue.png` 256×256 (sha16 `0353ce327c8c41b4`/`bd9a2bf1acad4be3`); guarded-запуск через `run_variant.sh` (layer, ts 1,1,1,0, mmproj→CUDA3, q4_0 KV, 131k, spec none, свежий `kvcache-v3t2-gate`). Bench-фаза на одном PID 208157 (:8081, старт 14:33:15): text1 «Синее» (prompt 27 tok), img_red «Красный квадрат.» (90 tok), img_blue «Синий квадрат.» (90 tok), text2 «Зелёная» (23 tok) — все HTTP 200, `finish_reason=stop`, использован `chat_template_kwargs.enable_thinking=false`; вердикт gate_ok=1. Peak VRAM GPU3 за image-фазу = 1037 МиБ (сэмплы 1/с); в журнале `loaded multimodal model` + предупреждение форка о минимум 1024 image tokens (зафиксировано). Новых Xid/AER = 0 с cursor; restore_done 14:39:10 health=200 llama_count=2; `bonsai-2` вернулся на :8082=200 в течение минуты после restore. OOM не было — коррекция долей/-ub не потребовалась. Замечание: run_variant на время варианта останавливает и bonsai-2 — coexistence проверяется отдельным guarded-тестом задачи 6. Артефакты: `snapshots/vision3-task3/` (gate-check.sh, gate-stdout.txt, gate_text*.json, gate_img_*.json, gate_imgbody_*.json, gate-verdict.txt=OK, vram3-samples.txt, vram-gate-before.txt, cursor-before-gate.txt, fix_red/fix_blue.png).

### Задача 4. Decode A/B на одном потоке при настоящем vision

**Результат:** выбранный режим decode прошёл с тем же самым проектором и q4-кэшем, что пойдут в службу.

- [~] Базовый arm `layer/no-MTP` с mmproj на GPU3, затем `tensor/no-MTP` — менять только split mode. С теми же model GGUF, projector, q4 K/V, pool 131072, batch/ubatch и картами сравнить `layer/MTP n3` против `layer/no-MTP`, `tensor/MTP n3` против `tensor/no-MTP`. Для MTP head оставить на GPU0–2; не переносить на медленный GPU3 ради мнимого выигрыша. n1/n3/n5 исследовать только если n3 реально выигрывает, проверять `draft_n`/`draft_accepted`.
- [~] Для каждого arm прогреть соответствующий prompt size и выполнить short/4k/32k/64k/около 120k, prose/code/architecture, фиксированный output 512+ с `ignore_eos`, минимум три повтора финалистов. Записать настоящие `prompt_n`, `prompt_tps`, `decode_tps`, `finish_reason`, TTFT, draft accepted, GPU memory/SM clock/power и код HTTP. Сравнивать медианы при одинаковой форме запросов, не переносить прошлые q8/layer цифры как baseline. Повторить начальный контроль в конце, отсеять температурный drift; не принять короткий тест за ответ о 120k.
- [x] После победителя повторить image-тест Qwen (не только текст). Две последовательные независимые near-limit заявки при пуле 131072 могут упереться в удержанный KV (`deferred tasks 1`): проверить и документировать. Не увеличивать контекст сверх заданного без отдельного решения; если stuck — это ограничение, а не успешный профиль.

### Задача 5. Один прежний API для текстовых и графических запросов Qwen

**Результат:** конечному клиенту не нужны второй URL/ключ и выбор иной модели для картинки.

- [x] Open WebUI хранит провайдера Qwen на приватном `:8081/v1` и пользовательские ключи в `webui.db`. Не пересоздавать БД/контейнер и не печатать ключи. Проверить сохранение прежнего model ID и объявление vision capability в WebUI после новой загрузки Qwen; при необходимости исправлять только metadata/параметры этого же model ID через поддерживаемый API после snapshot, без публикации backend-key.
- [x] С ОДНИМ существующим пользовательским `sk-...` и тем же публичным `/api/chat/completions` отправить (а) текст, (б) data-URL картинку+вопрос, с одинаковым `model=Qwen3.8-27B`, в режимах `stream:false` и `stream:true`. На image ждать корректного description и `[DONE]` в SSE, `finish_reason=stop`; неверный ключ → 401. По `llama-qwen` PID/journal проверить, что оба попали в ТОТ ЖЕ процесс; случайный ответ Bonsai или другой model ID — FAIL.
- [x] Если пользователи также выбирают другие явно названные llama-модели: оставить им их прежние ID и второй provider, но **не** перехватывать Qwen image. Проверить, что Open WebUI уже разделяет запросы по выбранному model ID, и признать, что балансировка одной Qwen-модели между несколькими llama здесь не обеспечивается. Отдельный router проектировать только при подтверждённой потребности и дополнительных GPU/VRAM, не устанавливать его в этой задаче.
- [x] Проверить 32k text Qwen и короткий image Qwen последовательно и при конкуренции клиентов на одном слоте: они могут ждать друг друга, поскольку `--parallel 1`. Фиксировать очередь/latency и лимит выдачи на стороне клиента; не выдавать один поток за параллельное обслуживание. Тесты из реального клиентского API отдельно от direct-backend тестов, чтобы измерить overhead WebUI.

### Задача 6. Конфликт GPU3, принятие и откат

**Результат:** Qwen image работает на GPU3 без тайного OOM и старый API сохранён.

- [x] Текущее `bonsai-2.service` использует около 8891 МиБ из 10240 МиБ. После замера Qwen projector+image peak провести ОТДЕЛЬНЫЙ guarded-тест одновременного Bonsai + Qwen image/decode; если запаса под compute нет, не оставлять Bonsai работающим на GPU3 при Qwen-vision — документировать потерю второй модели/порта :8082 и убедиться, что Open WebUI не предлагает неработающий Bonsai. Картинки Qwen не перенаправлять в Bonsai даже как fallback.
- [x] Сохранить в `profiles/qwen38-vision3-131k.service` точный победивший набор аргументов и environment (`LD_LIBRARY_PATH`, CUDA visibility, `GGML_CUDA_TURING_CUBLAS_MIN_M=256`), `systemd-analyze verify`; подготовить rollback текущего эталонного unit и WebUI metadata. При применении на стенде — `daemon-reload` ДО restart, затем read-back установленного `ExecStart`, live PID/cmdline, версия бинарника, placement VRAM, обе разновидности ответа через старый клиентский URL и прежний ключ, Xid с cursor. Обновить единственный бэкап актуального эталонного unit после принятия; при любой ошибке вернуть исходные unit/metadata и оба исходно активных сервиса.
- [x] Программно подсчитать ожидаемые/фактические JSONL строки, отфильтровать non-200/незавершённые и занести медиану/range decode+prefill по arm+case, VRAM четырёх карт, image payload fixture и ответы без секретов, лимит последовательных long requests, решение по Bonsai и точный rollback в `summaries/qwen38-vision3-131k.md`. На телефон — узкие блоки до ~45 символов, без широкой таблицы.

**Критерий окончания:** один и тот же Qwen PID на GPU0–2 с mmproj на GPU3 успешно отвечает на text и image с прежним `model`, URL и пользовательским ключом; KV q4_0/131072, single-stream decode измерен с работающим vision; конфликт Bonsai решён и откат проверен. Если любой gate не пройден — сохранить прежний сервис и явно назвать блокер, а не подменять картинку ответом другой модели.

## Статус выполнения (24.09, 17:25)

Новый профиль применён на стенде и проверен end-to-end.

- Юнит `qwen38-vision3-131k.service`: **active + enabled**, `:8081`, PID 219173.
- `--mmproj-device CUDA3`, `--parallel 1 --ctx-size 131072`, KV q4_0, `layer`, `ts 1,1,1,0`, `spec none`.
- VRAM: GPU0 6933 / GPU1 7575 / GPU2 8203 (из 20480), GPU3 1029 (из 10240) МиБ.
- Прежний клиентский путь (WebUI `:4000`, один `sk-…`, `model=Qwen3.8-27B`): текст 200 «Голубое», image 200 «Синий», SSE `stream:true` c `[DONE]`, неверный ключ 401 — всё на одном PID.
- Профиль сохранён: `profiles/qwen38-vision3-131k.service` (sha256 `aaf207158194f8d0…`); отчёт: `summaries/qwen38-vision3-131k.md`.
- Старые `llama-qwen.service` и `bonsai-2.service` остановлены и **disabled** (до отдельной команды); бэкап эталона — `backups/llama-qwen.service.20260924` + `.d`.
- Xid/AER с курсора: 0.

Незавершённое (задача 4): arm `layer-nomtp` дал 13 строк, но p120k встал на `deferred tasks 1` (queue stuck ~1 ч 50 мин) и был снят вручную; arms `tensor-nomtp`, `layer-mtp3`, `tensor-mtp3` не запускались. Итоговые цифры decode по всем режимам пока не получены — они нужны для выбора финального профиля при возврате к A/B.

