# Strata на 4× CMP 50HX (ai100gb / 192.168.50.9) — установка, патч, правила запуска, бенчмарки

Состояние на 2026-10-05. Секретов (пароли, токены, кошельки) здесь нет намеренно.

Стенд: тот же, что в llama.cpp-кампаниях этого репозитория — три модифицированные CMP 50HX (TU102, sm_75) + стоковая 10 GB.
Движок: **Strata** (Niko1221/Strata), специализированный под Qwen3.8-Flash-Next (125B MoE, 6B активных).

---

## 1. Железо

| Компонент | Что стоит |
|---|---|
| GPU | 4× NVIDIA CMP 50HX (TU102, **sm_75 / Turing**), все compute-unlocked драйвером **610.43.03** (`xrip/cmp50hx-unlock`) |
| | GPU0/1/2 — **20 GB** (перепаяны 16 Gbit GDDR6), PCIe 01:00.0 / 02:00.0 / 04:00.0, **Gen2 x8** |
| | GPU3 — стоковая **10 GB** (05:00.0), **Gen2 x4** |
| BIOS | **Above 4G decoding — ВЫКЛ** (иначе unlock-драйвер падает `NV_ERR_MEMORY_ERROR 0x72`) |
| CPU | Xeon E5-2620 v1 (6C/12T, **AVX, без AVX2**) → движок собирается с `STRATA_ISA_FLOOR=avx` |
| RAM | 15 GB + swap 4 GB (эксперты держатся в VRAM — RAM хватает) |
| Диск | `/mnt/usbsata` 894 GB (SATA SSD): модели, паки, install root |

---

## 2. Установка Strata

```bash
cd /mnt/usbsata
git clone https://github.com/Niko1221/Strata.git strata && cd strata

# llama.cpp на пиннутом коммите (как в setup.py: LLAMA_CPP_COMMIT)
curl -L -o /tmp/lc.zip https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip
mkdir -p third_party && unzip -q /tmp/lc.zip -d third_party/_u
mv third_party/_u/llama.cpp-3cf0325* third_party/llama.cpp && rmdir third_party/_u

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Сборка движка (проверена на этом стенде):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_ISA_FLOOR=avx \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target strata -j 4
```

**Грабли:**

- **CUDA 13 не подходит**: в установленном комплекте нет cuBLAS →
  `Target "strata_prefill" links to CUDA::cublas, but the target was not found`. Берём **CUDA 12.8** (`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`).
- Нет AVX2 → обязателен `-DSTRATA_ISA_FLOOR=avx` (официальный экспериментальный путь для Sandy/Ivy Bridge). CPU-пул при этом медленный → стратегия «все эксперты в VRAM».
- Инкрементальная пересборка после правок: `cmake --build build --target strata -j 4` (~30 с).

---

## 3. Модели и паки

| Модель | GGUF | Эксперты | Пак | Контекст в конфиге |
|---|---|---|---|---|
| Coder IQ1_M | `/mnt/usbsata/models/strata-coder/` | 12 288 | `packs/coder` | 65 536 |
| Полная Q2_0 | `/mnt/usbsata/models/strata-qwen/` | 24 576 | `packs/qwen20` | 262 144 |

```bash
M=/mnt/usbsata/models/strata-qwen/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
.venv/bin/python tools/iq_pack.py --gguf $M --out packs/qwen20
.venv/bin/python tools/iq_pack.py --gguf $M --out packs/qwen20 --experts-bin
.venv/bin/python tools/strata_tokenizer.py --gguf $M --out packs/qwen20

# MTP-слой (общий для обеих моделей; кладётся в data/mtp/rt)
.venv/bin/python tools/mtp_fetch.py fetch --out data/mtp
.venv/bin/python tools/mtp_pack.py --src data/mtp --experts q2_0 --out data/mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf data/mtp/mtp-q2_0.gguf --out data/mtp/rt
```

**Грабли:**

- `tools/pack_index.py` на нативных IQ-паках падает (`no such file: packs/<name>/manifest.json`) — шаг не нужен, пропускается.
- **Фикс mmap-арены (обязателен).** При `STRATA_ARENA_MMAP=1` движок мапит `packs/<name>/experts.bin` read-only **только если размер файла точно равен `arena_want + один максимальный блоб`**. `iq_pack --experts-bin` пишет файл короче ровно на один блоб → движок молча уходит в pinned-арену и умирает на `MAP_HUGETLB ... vm.nr_hugepages=0`. Лечение:
  - Coder: `dd if=/dev/zero bs=2662400 count=1 >> packs/coder/experts.bin`
  - Q2_0: `truncate -s 33975244800 packs/qwen20/experts.bin` (было 33 973 862 400)
- Слой `draft_vocab.bin` для MTP: положить `data/draft_vocab_cyrillic.bin` (кириллица + код) в `data/mtp/rt/draft_vocab.bin`, если нужны русские ответы.

---

## 4. Правила запуска (systemd)

Работает **один** инстанс Strata: модель занимает три 20-GB карты целиком.

| Юнит | Модель | Порт | Состояние |
|---|---|---|---|
| `strata-qwen.service` | полная **Q2_0** | `0.0.0.0:8080` | **active, enabled** |
| `strata-coder.service` | **Coder IQ1_M** | `0.0.0.0:8080` | disabled (альтернатива) |

Переключение модели: `sudo systemctl stop strata-qwen && sudo systemctl disable strata-qwen && sudo systemctl enable --now strata-coder` (и наоборот).

**Важно:** порт/хост задаются **и** в JSON, **и** в `ExecStart` юнита — `serve/server.py --port N` **перекрывает** конфиг. Править оба места + `systemctl daemon-reload`. Первый старт после рестарта — 2–4 минуты (загрузка пака, прогрев кэша экспертов), порт появляется не сразу: проверять `ss -tln | grep 8080`, а не считать это ошибкой.

Файлы: `docs/strata-qwen.json`, `docs/strata-coder.json`, `systemd/strata-*.service`.

Ключевое в конфиге Q2_0:

```json
"gpu": [0, 1, 2], "layer_split": "16,32", "parallel": 2,
"args": ["--pack", "packs/qwen20", "--native", "<shard1>", "--ple-gguf", "<shard2>",
         "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
         "--trim-stage-weights", "--prefill", "2048", "--spec", "4", "--spec-min-p", "0.5",
         "--mtp", "data/mtp/rt", "--max-context", "262144", "--kv", "int8",
         "--vram-reserve-mib", "600", "--pcie-frac", "0"],
"env": {"STRATA_ARENA_MMAP": "1", "STRATA_BF16_TC": "1", "STRATA_STAGE_TRIM": "1"},
"fit_max_tokens": true, "aliases": "strata,coder,qwen3-coder",
"reasoning_budget_tokens": 8192, "anthropic_thinking": "on_request",
"effort_position": "end", "api_monitor": true, "engine_silence_s": 600,
"sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 32, "min_p": 0.02}
```

Пояснения к настройкам:

- `STRATA_BF16_TC=1` — BF16-проекции через FP16 tensor cores Turing (#655, на 2080 Ti давал +15–18 % префилла).
- `effort_position: end` — смена reasoning_effort не рушит prefix-кэш.
- `fit_max_tokens: true` — большой `max_tokens` урезается до остатка контекста вместо 400.
- `sampling` — заданные по умолчанию сэмплинг-параметры для клиентов, которые своих не присылают (пустой блок = greedy, что лучше для скорости MTP-спекуляции).
- `engine_silence_s: 600` — таймаут «движок молчит» поднят с 300 с (долгие чтения 100K+ промптов на этом железе).

**Итог загрузки Q2_0:** эксперты **100 % в VRAM** (8 192 слота на карту × 3 = все 24 576), VRAM 17.3 / 17.3 / 17.8 GB из 20.5, RAM свободно ~10 GB, VRAM-запас ~2.7 GB.

---

## 5. Патч движка: `parallel ≥ 2` + 100 % резидентность экспертов

**Симптом.** Первый же параллельный запрос → `verify batch: timed out at layer 0` → движок выходит с кодом 1 (`the engine stopped unexpectedly`); сервер отвечает двум клиентам ошибками. Соло-запросы при этом работают идеально.

**Причина (upstream #776, #792, #845; на 2026-10-05 в main не влито).** Когда стадия держит все свои эксперты в VRAM, её окно верификации пишется как «zero-doorbell» граф (#646): `record_window` не публикует по-слойный `m_seq_`. Соло-путь (`Verifier::run`) это учитывает веткой `all_resident_`, а batch-хост-пути — нет: `run_slot_rows` (без `--batch-groups`) и `batch_poll` (с ним) крутятся в `while (*seq < want)` до 20-секундного таймаута. Воспроизводится на любом контексте (проверено 32K и 256K), в т.ч. с одной занятой строкой окна.

**Патч** — `patches/strata-batch-zerodoorbell-792.patch` (в `src/core/verify.cpp`, 11 строк):

```cpp
// run_slot_rows(): сразу после cudaGraphLaunch(exec_bm_[batch_key(rows, S, 0)], cs_) и перед per-layer циклом
if (all_resident_) {
    *flag = 1;      // PLE-строки уже собраны в stage_batch(); ждать будет cudaStreamSynchronize ниже
} else
for (int64_t k = 0; k < steps; ++k) { ... }

// batch_launch(): вместо b_k_ = 0;
b_steps_ = le_ - lb_;
if (all_resident_) *(volatile uint32_t*) h_flag_ = 1;
b_k_ = all_resident_ ? b_steps_ : 0;
```

Применение и пересборка:

```bash
cd /mnt/usbsata/strata
git apply /path/strata-batch-zerodoorbell-792.patch
cmake --build build --target strata -j 4        # ~30 с
sudo systemctl restart strata-qwen
```

**После `git pull` Strata патч накладывать заново**, пока #792/#845 не вмержат в main.

**Проверено после патча:** 2 параллельных запроса по 200 токенов — оба успешны за **8.7 с**; 3 параллельных по 150 токенов — все успешны; `grep -c "never rang|timed out at layer"` = 0; `GET /v1/status` → `concurrency: {serving: 2, requested: 2}`.

---

## 6. Бенчмарки (майнер выключен, часы ~1900 МГц)

### Полная Q2_0 (3×20 GB, 100 % экспертов в VRAM)

| Метрика | 131K, 1 запрос | 256K + `parallel: 2` (с патчем) |
|---|---:|---:|
| Decode, короткий промпт (256 ток.) | **80.2–81.8 tok/s** | 70.4 tok/s |
| Decode на 30K | 81.8 tok/s | — |
| Decode на 100K | **62.0 tok/s** | — |
| Prefill 6.5K | 1 736 tok/s | — |
| Prefill 30K | 2 366 tok/s | — |
| Prefill 61K | 2 401 tok/s (TTFT 25 с) | — |
| Prefill 101K | **2 082 tok/s** | — |
| Reuse чекпоинтов | 101 257 из 101 300 токенов за **0.58 с** | — |
| 2 параллельных × 200 ток. | — | **8.7 с** на оба |
| 3 параллельных × 150 ток. | — | ок, 0 ошибок |

### Coder IQ1_M (3×20 GB, контекст 65K)

| Метрика | Значение |
|---|---:|
| Decode (256 ток.) | 63–69 tok/s |
| Decode на 30K | 60.3 tok/s |
| Prefill 6.5K | 1 676 tok/s |
| Prefill 30K | 2 520 tok/s |

### Влияние майнера

При `rgminer` на всех четырёх картах decode падает примерно вдвое (Coder: 26–29 tok/s). Все цифры выше — с выключенным майнером.

---

## 7. Сосуществование с rgminer (pearl@kryptex)

- Юнит `rgminer.service` (root, `Restart=on-failure`) держит ~4.9 GB VRAM на карту; его OOM-guard требует свободных **~6.5 GB на карту**.
- При полной Q2_0 (17–18 GB на карту) на GPU0–2 места нет → `GPU context disabled reason=oom` (`ampere_table.dPackedB`) → crash-loop юнита.
- Рабочая схема: drop-in `systemd/rgminer-gpu3-only.conf`:

```ini
[Service]
Environment=CUDA_VISIBLE_DEVICES=3
```

  майнер работает только на стоковой 10-GB карте (GPU3), Strata остаётся на полной скорости.
- Управление: `sudo systemctl stop rgminer` / `sudo systemctl start rgminer`.

---

## 8. Быстрые проверки

```bash
curl -s http://192.168.50.9:8080/v1/models       # id + aliases
curl -s http://192.168.50.9:8080/v1/status       # concurrency.serving, context.native
curl -s http://192.168.50.9:8080/api-monitor     # последние 100 запросов (api_monitor: true)
nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv
grep -iE "100% of the experts resident|batch windows|timed out|never rang" /mnt/usbsata/strata/strata-qwen.log | tail
```

---

## 9. Выводы

1. Полной Qwen3.8-Flash-Next (125B MoE) **хватает 60 GB VRAM трёх разблокированных CMP 50HX**: все 24 576 экспертов живут в VRAM, CPU-пул почти не используется, RAM-маппинг арены работает при 15 GB RAM.
2. `sm_75` (Turing) официально поддержан движком — портировать ядра не нужно; полезны `STRATA_BF16_TC=1` и сборка без AVX2 (`STRATA_ISA_FLOOR=avx`).
3. 256K-контекст и 2 параллельных запроса **рабочие, но требуют локального патча** (§5) — без него любой параллельный запрос убивает движок.
4. Ограничители на этом стенде: 15 GB RAM (старт/страницы mmap-арены), отсутствие AVX2 (медленный CPU-пул → держать всё в VRAM), PCIe Gen2 x8 (при 100 % резидентности не критично).
5. Майнер и полная модель вместе в 60 GB не живут — либо майнер на 10-GB карте, либо модель на трёх.