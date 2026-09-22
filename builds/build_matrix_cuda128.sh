#!/usr/bin/env bash
set -u
SRC=/home/fastchip/llama-fork-build/source
ROOT=/home/fastchip/llama-fork-build/research
MAN=$ROOT/manifest.jsonl
mkdir -p "$ROOT"
build_one(){
 label=$1; shift; b=$ROOT/$label; rm -rf "$b"; start=$(date -u +%FT%TZ)
 cmake -S "$SRC" -B "$b" -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@" >"$b.configure.log" 2>&1
 crc=$?
 if [[ $crc -eq 0 ]];then cmake --build "$b" --target llama-server llama-bench -j 16 >"$b.build.log" 2>&1; brc=$?;else brc=99;fi
 end=$(date -u +%FT%TZ); commit=$(git -C "$SRC" rev-parse HEAD); nvcc=$(/usr/local/cuda-12.8/bin/nvcc --version|grep release|sed 's/"/\\"/g');
 if [[ $brc -eq 0 ]];then
   sha=$(sha256sum "$b/bin/llama-server"|cut -d' ' -f1); cache=$(sha256sum "$b/CMakeCache.txt"|cut -d' ' -f1); cc=$(sha256sum "$b/compile_commands.json"|cut -d' ' -f1); ldd "$b/bin/llama-server" >"$b.ldd.txt"; status=PASS
 else sha=;cache=;cc=;status=FAIL;fi
 printf '{"label":"%s","start":"%s","end":"%s","commit":"%s","configure_rc":%d,"build_rc":%d,"status":"%s","server_sha256":"%s","cmake_cache_sha256":"%s","compile_commands_sha256":"%s","nvcc":"%s"}\n' "$label" "$start" "$end" "$commit" "$crc" "$brc" "$status" "$sha" "$cache" "$cc" "$nvcc" >>"$MAN"
}
: >"$MAN"
build_one cuda128-nccl-on-lto-off -DGGML_CUDA_NCCL=ON -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda128-nccl-off-lto-off -DGGML_CUDA_NCCL=OFF -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda128-nccl-on-lto-on -DGGML_CUDA_NCCL=ON -DGGML_LTO=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda128-nccl-on-fa-all -DGGML_CUDA_NCCL=ON -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON
