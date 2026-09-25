# Qwen3.8 Two-GPU Balance and Throughput Implementation Plan

> **For agentic workers:** Execute task-by-task with a single durable campaign owner. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and deploy a balanced two-GPU Qwen3.8 profile that maximizes PP and decode while retaining a verified 256K request and usable two-client throughput.

**Architecture:** A durable stand-side campaign runs isolated one-variable arms through the existing rollback-safe `run_variant.sh`. JSONL rows and per-arm telemetry are reduced after each phase; later phases use only the selected split/ubatch. The installed service remains the rollback reference until the final acceptance gate passes.

**Tech Stack:** bash, systemd-run, Python stdlib benchmark clients, llama.cpp upstream a02c7f5, CUDA 12.8, NVIDIA CMP 50HX.

## Global Constraints

- Use exactly GPU0 and GPU1 for Qwen during this campaign.
- Keep model, projector, context 262144, parallel 2, unified Q8 KV, MTP n=3, batch 2048, MMQ/cuBLAS threshold 256, FlashAttention, sampling, and CPU threads fixed unless the phase explicitly changes one of them.
- Never run two variant runners concurrently.
- Restore and verify `llama-qwen.service` after every arm.
- Reject new CUDA OOM/Xid/AER events and incomplete/non-200 benchmark rows.
- Preserve all raw rows and logs under a unique campaign stamp.

---

### Task 1: Campaign harness and baseline capture

**Files:**
- Create: `campaigns/qwen38-2gpu-balance-campaign.sh`
- Create: `harness/qwen38_2gpu_select.py`
- Create: `harness/qwen38_2gpu_report.py`
- Test: `harness/test_qwen38_2gpu_select.py`

**Interfaces:**
- Consumes: existing `harness/run_variant.sh`, `harness/bench_request.py`, `harness/bench_concurrency.py`.
- Produces: append-only request/concurrency JSONL, `state/2gpu-selection.json`, and a markdown report.

- [ ] Capture the live unit, PID/cmdline, binary version, model metadata, `/slots`, per-GPU VRAM, PCIe links, health, and a kernel-journal cursor.
- [ ] Add selector fixture tests proving that failed/incomplete rows are rejected and that the lowest maximum VRAM arm wins only when PP/decode gates pass.
- [ ] Run the selector tests with `python3 -m unittest -v harness/test_qwen38_2gpu_select.py`; require all tests PASS.
- [ ] Preflight every generated argv against `llama-server -m /nonexistent.gguf`; require only the expected missing-model failure and no unknown/duplicate-option warning.
- [ ] Smoke-test the restore path with a no-op short arm; require reference health 200, exact restored unit hash, one listener, and zero new Xid.

### Task 2: Layer-placement phase

**Files:**
- Modify: `campaigns/qwen38-2gpu-balance-campaign.sh`
- Output: `raw/2gpu-balance-<stamp>.jsonl`
- Output: `logs/2gpu-balance-<stamp>/split-*.log`

**Interfaces:**
- Produces selected `tensor_split` in `state/2gpu-selection.json`.

- [ ] Run same-window arms `1.00,1.00`, `1.10,0.90`, `1.15,0.85`, `1.20,0.80`, each at ubatch 512 and p32k for prose/code/architecture.
- [ ] Record pre/post VRAM, PP, decode, MTP acceptance, clocks, power, HTTP status, and post-cursor GPU errors for every row.
- [ ] Reject arms with max free VRAM below 2 GiB at screening, PP/decode below 98% of same-run baseline, or any failed row.
- [ ] Run p120k prose once on the two best surviving splits.
- [ ] Select the split minimizing maximum used VRAM; break ties within 256 MiB by higher p120k PP, then decode.

### Task 3: Ubatch pipeline phase

**Files:**
- Modify: `campaigns/qwen38-2gpu-balance-campaign.sh`
- Output: same stamped JSONL/log directory.

**Interfaces:**
- Consumes selected split.
- Produces selected `ubatch` in `state/2gpu-selection.json`.

- [ ] Run ubatch 256, 320, 384, and 512 with batch fixed at 2048.
- [ ] For each arm run short, p4k, and p32k across all three prompt classes with two repetitions.
- [ ] Compute median PP/decode per class and reject half-speed/degraded-server signatures before aggregation.
- [ ] Gate the best two Pareto candidates at p120k prose once.
- [ ] Select highest p120k PP subject to p32k decode and two-client acceptance gates; if PP differs by less than 2%, choose the lower-VRAM ubatch.

### Task 4: Layer versus tensor execution mode

**Files:**
- Modify: `campaigns/qwen38-2gpu-balance-campaign.sh`
- Output: same stamped JSONL/log directory.

**Interfaces:**
- Consumes selected split and ubatch.
- Produces selected `split_mode`.

- [ ] Run layer/pipelined and tensor arms with MTP n=3 fixed.
- [ ] Measure solo p32k and p120k across all prompt classes.
- [ ] Measure two simultaneous distinct prompts for three batches and record aggregate and per-request decode.
- [ ] Select a Pareto winner: maximize p120k PP, with solo decode and two-client aggregate no worse than 95% of the corresponding best/reference values.

### Task 5: Near-limit and client-path acceptance

**Files:**
- Create: `summaries/QWEN38-2GPU-BALANCE-<stamp>.md`
- Update: `docs/RESUME.md`
- Create: `profile/llama-qwen-2gpu-balanced.service`

**Interfaces:**
- Consumes final split/mode/ubatch.
- Produces promotable unit and verified report.

- [ ] Run one calibrated 256000-token prompt plus 128 generated tokens; require HTTP 200, complete timings, no OOM/Xid, and at least 1.5 GiB free on each active GPU.
- [ ] Run three two-client batches with distinct prompts; require both requests HTTP 200 and aggregate throughput within the gate.
- [ ] Run text, known-colour vision, SSE `[DONE]`, and wrong-key 401 probes through the actual client path.
- [ ] Aggregate and programmatically verify expected row counts before writing the report.
- [ ] Promote the winner only if every gate passes; otherwise leave the current two-GPU unit unchanged.
- [ ] Read back unit contents, live PID/cmdline, `/slots`, GPU occupancy, health, and binary version. Record rollback unit hash and exact raw-data paths.

### Task 6: Publish reproducible evidence

**Files:**
- Add stamped raw rows, logs, summary, selection JSON, campaign script, and final profile to the private research repository.

- [ ] Secret-scan staged files for credentials/tokens/passwords.
- [ ] Commit design/plan separately from measured results.
- [ ] Push using the temporary credential-helper procedure.
- [ ] Verify the remote commit SHA via the GitHub API and record it in the summary.
