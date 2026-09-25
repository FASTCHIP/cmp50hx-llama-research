# Qwen3.8 Two-GPU Balance and Throughput Design

## Goal

Balance Qwen3.8-27B layer placement across GPU0/GPU1, then maximize prompt processing and decode while preserving one 256K request and usable two-client throughput.

## Fixed reference profile

- Host: `192.168.50.9`, GPUs 0 and 1 only, both CMP 50HX 20 GiB.
- Binary: `/home/fastchip/llama.cpp-upstream-mmq-a02c7f5/bin/llama-server`.
- Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, projector present.
- `--ctx-size 262144 --parallel 2 --kv-unified --kv-unified-per-slot 262144`.
- Q8_0 K/V cache, FlashAttention on, MTP n=3, batch 2048.
- Current reference: layer split `1,1`, ubatch 512.
- Verified p250k: 256006 prompt tokens, PP 249.49 tok/s, decode 16.11 tok/s, HTTP 200, no CUDA OOM/Xid.
- Reference peak VRAM after p250k: GPU0 15587 MiB, GPU1 18759 MiB.

## Design

Use staged same-run A/B tests. Change one axis per phase and restore the exact reference unit after every arm.

1. Layer-balance phase: compare tensor splits `1,1`, `1.10,0.90`, `1.15,0.85`, `1.20,0.80` with all other parameters fixed. Screen at p32k, then gate the two best at p120k. Select the arm that minimizes maximum per-GPU VRAM subject to PP/decode remaining within 2% of reference and zero new CUDA/Xid errors.
2. Pipeline microbatch phase: on the selected split compare ubatch 256, 320, 384, 512. Keep batch at 2048. Screen short/p4k/p32k and gate the best candidates at p120k.
3. Execution-mode phase: compare layer/pipelined and tensor split using the winning ubatch, with MTP fixed. Measure solo p32k/p120k and two-client aggregate throughput.
4. Final gate: run p250k plus 128 generated tokens, then two distinct simultaneous requests. Promote only a Pareto winner that keeps p250k and dual-client service healthy.

## Acceptance criteria

- No `cudaMalloc failed`, CUDA OOM, Xid, uncorrected AER, HTTP failure, incomplete timing row, or restore failure.
- At least 1.5 GiB free on each active GPU after the p250k gate; target at least 2 GiB during screening.
- Prefer VRAM difference no greater than 1 GiB; accept a larger difference only when a layer/KV boundary makes the next split worse.
- p250k PP should improve; solo decode and two-client aggregate throughput may regress by no more than 5% from the same-run reference.
- Every claim uses rows from the current campaign, not archived values.

## Safety and recovery

Each arm snapshots `/etc/systemd/system/llama-qwen.service`, current service states, and a kernel-journal cursor. The runner stops the reference service, starts exactly one isolated test server, health-gates it, writes append-only JSONL, and restores the original unit in an unconditional EXIT trap. Any new Xid aborts the campaign. Final promotion requires systemd unit read-back, PID/cmdline verification, `/health` 200, text/vision/SSE/auth gates, and a near-limit request.
