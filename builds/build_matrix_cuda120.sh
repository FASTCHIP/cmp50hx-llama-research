#!/usr/bin/env bash
# CUDA 12.0 build matrix — same 4 variants as the CUDA 12.8 matrix.
# Runs inside CT 200 (vllm-build) on the Proxmox host yush-proxnew2.
set -u
SRC=/home/fastchip/llama-fork-build/source
ROOT=/home/fastchip/llama-fork-build/research
MAN=$ROOT/manifest-cuda120.jsonl
NVCC=/usr/bin/nvcc                     # CUDA 12.0 V12.0.140
CFLAGS='-march=sandybridge -mtune=sandybridge -O3 -fcf-protection=none'
: >"$MAN"

echo "nvcc: $($NVCC --version | grep release)"
echo "src commit: $(git -C "$SRC" rev-parse HEAD)"

build_one(){
  label=$1; shift
  b=$ROOT/$label; rm -rf "$b"; start=$(date -u +%FT%TZ)
  cmake -S "$SRC" -B "$b" -G Ninja \
    -DCMAKE_CUDA_COMPILER="$NVCC" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_BMI2=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "-DCMAKE_C_FLAGS=$CFLAGS" "-DCMAKE_CXX_FLAGS=$CFLAGS -Wno-changes-meaning" \
    "$@" >"$b.configure.log" 2>&1
  crc=$?
  if [ "$crc" -eq 0 ]; then
    cmake --build "$b" --target llama-server llama-bench -j 16 >"$b.build.log" 2>&1
    brc=$?
  else
    brc=99
  fi
  end=$(date -u +%FT%TZ)
  commit=$(git -C "$SRC" rev-parse HEAD)
  if [ "$brc" -eq 0 ]; then
    sha=$(sha256sum "$b/bin/llama-server" | cut -d' ' -f1)
    cache=$(sha256sum "$b/CMakeCache.txt" | cut -d' ' -f1)
    cc=$(sha256sum "$b/compile_commands.json" | cut -d' ' -f1)
    ldd "$b/bin/llama-server" >"$b.ldd.txt"
    # CET endbr64 must be absent: the serving CPU is Sandy Bridge (no CET).
    endbr=$(objdump -d "$b/bin/llama-server" 2>/dev/null | grep -c endbr64)
    status=PASS
  else
    sha=; cache=; cc=; endbr=-1; status=FAIL
  fi
  printf '{"label":"%s","start":"%s","end":"%s","commit":"%s","nvcc":"%s","configure_rc":%d,"build_rc":%d,"status":"%s","server_sha256":"%s","cmake_cache_sha256":"%s","compile_commands_sha256":"%s","endbr64_count":%s,"cpu_flags":"sandybridge no-avx2 no-fma no-f16c no-cet"}\n' \
    "$label" "$start" "$end" "$commit" "$($NVCC --version | grep -o 'release [0-9.]*')" "$crc" "$brc" "$status" "$sha" "$cache" "$cc" "$endbr" >>"$MAN"
  echo "done $label status=$status build_rc=$brc endbr64=$endbr"
}

build_one cuda120-nccl-on-lto-off  -DGGML_CUDA_NCCL=ON  -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda120-nccl-off-lto-off -DGGML_CUDA_NCCL=OFF -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda120-nccl-on-lto-on   -DGGML_CUDA_NCCL=ON  -DGGML_LTO=ON  -DGGML_CUDA_FA_ALL_QUANTS=OFF
build_one cuda120-nccl-on-fa-all   -DGGML_CUDA_NCCL=ON  -DGGML_LTO=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON

echo "MATRIX COMPLETE"
cat "$MAN"
