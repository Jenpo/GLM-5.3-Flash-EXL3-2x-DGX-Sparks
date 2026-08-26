#!/bin/bash
# GLM-5.3-Flash (native FP8) on vLLM, TP4, detached under --restart unless-stopped.
#
# Weights come from the local ~306 GiB checkout mounted at /model, not from the repo id
# zai-org/GLM-5.3-Flash: passing the repo id would re-download the whole checkpoint.
#
# The image is the local overlay from patches/Dockerfile.glm53-flash-sm120, which pads
# this checkpoint's NoPE MLA (qk_rope_head_dim=0) into the 576-wide GLM_NSA geometry.
# Stock vllm/vllm-openai:glm53-flash dies on the first forward with
# "pe_dim must be 64 for fp8_ds_mla"; see that Dockerfile for the full reasoning.
#
# On SM120 the only sparse-MLA backend is FLASHINFER_MLA_SPARSE_SM120 (cuda.py returns
# [TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120] for capability major 12), and its impl only
# accepts the packed fp8_ds_mla KV layout, which vLLM derives from auto/fp8/fp8_e4m3.
# KV_CACHE_DTYPE=bfloat16 therefore has no sparse decode kernel here.
#
# 4x RTX PRO 6000 (SM120, 96 GiB) leaves ~14 GiB/GPU after the ~306 GiB FP8 weights at
# TP4, so MAX_NUM_SEQS is low: the 34 KDA layers allocate a per-sequence recurrent-state
# slot that scales with concurrency, not with context length.
#
# --privileged is deliberately not used; --gpus all + --ipc=host is what vLLM needs.
#
# Usage:
#   ./vllm-glm53-flash.sh
#   SPEC_TOKENS=0 ./vllm-glm53-flash.sh
#   docker logs -f vllm-glm53-flash

set -euo pipefail

IMAGE="${IMAGE:-cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/GLM-5.3-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-glm53-flash}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
# FlashInfer's SM120 sparse-MLA routes decode batches above _DECODE_MAX_TOKENS=64 to the
# general prefill orchestrator instead of the specialized split-K decode kernel, and MTP
# makes each sequence contribute SPEC_TOKENS+1 tokens per step, so 10 * (5 + 1) = 60
# keeps decode on the specialized kernel.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Context costs ~8.7 KiB/token of KV (656 B MLA record + indexer + KDA-aligned pages), so
# 524288 tokens needs 4.35 GiB for a single max-length request. The MTP draft weights are
# inside the profiled budget, which leaves little room: 0.93 yielded only 2.76 GiB of KV
# (max usable context 318,976) and failed to start. The DSA indexer also allocates
# max_model_len * 40 * 132 B (2.6 GiB at 512k) *after* profiling, so it eats the physical
# slack above this fraction rather than the KV pool -- at 0.95 that slack is ~2.4 GiB.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
# vLLM reserves an estimate for the CUDA graph pool (0.84 GiB estimated vs 0.18 GiB
# actual here, i.e. 0.95 behaves like 0.9412). Disabling the estimate returns ~0.66 GiB
# to KV; the pool is still capped by the real capture.
ESTIMATE_CUDAGRAPHS="${ESTIMATE_CUDAGRAPHS:-0}"
# fp8 KV is the Blackwell path in the recipe and the only one the FlashInfer SM120
# sparse-MLA kernel takes; bfloat16 doubles the DSA KV footprint.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# The checkpoint ships one MTP layer (text_config.num_nextn_predict_layers=1); vLLM
# reuses it for each step, which is why acceptance drops as this grows. 5 matches the
# recipe's TP4 profile. Outside sm_10x the DeepGEMM fp8 paged MQA logits kernel only
# serves next_n in (1, 2), so anything above SPEC_TOKENS=1 puts the DSA indexer on its
# flattening fallback. SPEC_TOKENS=0 disables speculation entirely.
SPEC_TOKENS="${SPEC_TOKENS:-5}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-glm53-flash}"

if [ "$#" -gt 0 ]; then
    echo "Positional arguments are not supported; set the env knobs instead." >&2
    exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    printf 'Missing model config: %s/config.json\n' "${MODEL_DIR}" >&2
    exit 1
fi

if [ ! -f "${MODEL_DIR}/tokenizer_config.json" ]; then
    printf 'Checkpoint is incomplete (no tokenizer_config.json): %s\n' "${MODEL_DIR}" >&2
    exit 1
fi

FIRST_SHARD="$(ls "${MODEL_DIR}"/model-*-of-*.safetensors 2>/dev/null | head -n 1)"
if [ -z "${FIRST_SHARD}" ]; then
    printf 'No weight shards found: %s\n' "${MODEL_DIR}" >&2
    exit 1
fi
EXPECTED_SHARDS="${FIRST_SHARD##*-of-}"
EXPECTED_SHARDS="${EXPECTED_SHARDS%%.safetensors}"
ACTUAL_SHARDS="$(ls "${MODEL_DIR}"/model-*-of-*.safetensors | wc -l)"
if [ "$((10#${EXPECTED_SHARDS}))" -ne "${ACTUAL_SHARDS}" ]; then
    printf 'Checkpoint is incomplete: %s of %s shards present in %s\n' \
        "${ACTUAL_SHARDS}" "$((10#${EXPECTED_SHARDS}))" "${MODEL_DIR}" >&2
    exit 1
fi

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" = "true" ]; then
        printf 'Container is already running: %s\n' "${CONTAINER_NAME}" >&2
        exit 1
    fi
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

SPEC_ARGS=()
if [ "${SPEC_TOKENS}" -gt 0 ]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
fi

# torch.compile and FlashInfer JIT artifacts, so only the first boot pays for them.
mkdir -p "${CACHE_ROOT}"

printf '[%s] TP%s mtp=%s kv=%s seqs=%s len=%s util=%s name=%s port=%s\n' \
    "${CONTAINER_NAME}" "${TP_SIZE}" "${SPEC_TOKENS}" "${KV_CACHE_DTYPE}" \
    "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" "${GPU_MEMORY_UTILIZATION}" \
    "${SERVED_MODEL_NAME}" "${PORT}"

docker run \
    --name "${CONTAINER_NAME}" \
    --detach \
    --restart "${RESTART_POLICY}" \
    --init \
    --gpus all \
    --ipc=host \
    --shm-size 32g \
    -p "${PORT}:${PORT}" \
    --env HF_HUB_OFFLINE=1 \
    --env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
    --volume "${MODEL_DIR}":/model:ro \
    --volume "${CACHE_ROOT}":/root/.cache \
    "${IMAGE}" \
    /model \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    "${SPEC_ARGS[@]}"

printf '[%s] detached (%s); first boot compiles kernels. Follow: docker logs -f %s\n' \
    "${CONTAINER_NAME}" "${RESTART_POLICY}" "${CONTAINER_NAME}"
