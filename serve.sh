#!/bin/bash
# GLM-5.3-Flash (EXL3) on vLLM, GB10 / SM121. Detached under --restart unless-stopped.
#
# Weights are an EXL3 checkout mounted at /model, not NVFP4 and not the Hub
# repo id zai-org/GLM-5.3-Flash. The overlay image pads this checkpoint's NoPE
# MLA (qk_rope_head_dim=0) into the 576-wide GLM_NSA geometry. Stock
# vllm/vllm-openai:glm53-flash-arm64-cu130 dies on the first forward with
# "pe_dim must be 64 for fp8_ds_mla"; see the Dockerfile for the full reasoning.
#
# On SM12x the only sparse-MLA backend is FLASHINFER_MLA_SPARSE_SM120 (cuda.py
# returns [TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120] for capability major 12),
# and its impl only accepts the packed fp8_ds_mla KV layout, which vLLM derives
# from auto/fp8/fp8_e4m3. KV_CACHE_DTYPE=bfloat16 therefore has no sparse decode
# kernel here.
#
# EXL3 is the weight format. Do not pass --moe-backend marlin: that is the
# NVFP4 FLASHINFER_CUTLASS workaround on this arch and will not drive EXL3
# experts.
#
# One GB10 (121 GiB UMA) cannot hold a 320B EXL3 MoE plus KV. Default is TP=2
# over two Sparks (mp + --nnodes). Single-node TP=1 is an override for a
# checkpoint that actually fits.
#
# --privileged is deliberately not used; --gpus all + --ipc=host is what vLLM
# needs. Multi-node also needs --network host for NCCL.
#
# Usage:
#   ./serve.sh
#   NODE_RANK=1 NNODES=2 ./serve.sh
#   SPEC_TOKENS=0 ./serve.sh
#   docker logs -f vllm-glm53-flash

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/GLM-5.3-Flash-EXL3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-glm53-flash}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-2}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29521}"
# FlashInfer's SM12x sparse-MLA routes decode batches above _DECODE_MAX_TOKENS=64
# to the general prefill orchestrator instead of the specialized split-K decode
# kernel, and MTP makes each sequence contribute SPEC_TOKENS+1 tokens per step,
# so 10 * (5 + 1) = 60 keeps decode on the specialized kernel. GB10 UMA is
# tighter than 4x 96 GiB, so default concurrency is 4.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Context costs ~8.7 KiB/token of KV (656 B MLA record + indexer + KDA-aligned
# pages). 0.95 is the 4x96 GiB number; GB10 UMA "free" counts reclaimable page
# cache the driver will not give the KV slab. 0.87 is the working budget.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
ESTIMATE_CUDAGRAPHS="${ESTIMATE_CUDAGRAPHS:-0}"
# fp8 KV is the only layout the FlashInfer SM12x sparse-MLA kernel takes.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# The checkpoint ships one MTP layer (text_config.num_nextn_predict_layers=1).
# Outside sm_10x the DeepGEMM fp8 paged MQA logits kernel only serves next_n
# in (1, 2), so anything above SPEC_TOKENS=1 puts the DSA indexer on its
# flattening fallback. SPEC_TOKENS=0 disables speculation entirely.
SPEC_TOKENS="${SPEC_TOKENS:-5}"
QUANTIZATION="${QUANTIZATION:-exl3}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-glm53-flash}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

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

SHARD_COUNT="$(find "${MODEL_DIR}" -maxdepth 1 -name '*.safetensors' | wc -l)"
if [ "${SHARD_COUNT}" -eq 0 ]; then
    printf 'No weight shards found: %s\n' "${MODEL_DIR}" >&2
    exit 1
fi

INDEX="${MODEL_DIR}/model.safetensors.index.json"
if [ -f "${INDEX}" ]; then
    EXPECTED_SHARDS="$(python3 - "${INDEX}" <<'PY'
import json, sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text())
files = {v for v in index.get("weight_map", {}).values()}
print(len(files))
PY
)"
    if [ "${EXPECTED_SHARDS}" -ne "${SHARD_COUNT}" ]; then
        printf 'Checkpoint is incomplete: %s of %s shards present in %s\n' \
            "${SHARD_COUNT}" "${EXPECTED_SHARDS}" "${MODEL_DIR}" >&2
        exit 1
    fi
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

QUANT_ARGS=()
if [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "none" ]; then
    QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi
EAGER_ARGS=()
if [ "${ENFORCE_EAGER}" = "1" ]; then
    EAGER_ARGS=(--enforce-eager)
fi

DIST_ARGS=()
if [ "${NNODES}" -gt 1 ]; then
    DIST_ARGS=(
        --distributed-executor-backend mp
        --nnodes "${NNODES}"
        --node-rank "${NODE_RANK}"
        --master-addr "${MASTER_ADDR}"
        --master-port "${MASTER_PORT}"
    )
    if [ "${NODE_RANK}" -gt 0 ]; then
        DIST_ARGS+=(--headless)
    fi
fi

# torch.compile and FlashInfer JIT artifacts, so only the first boot pays for them.
mkdir -p "${CACHE_ROOT}"

printf '[%s] TP%s nnodes=%s rank=%s mtp=%s kv=%s quant=%s seqs=%s len=%s util=%s name=%s port=%s\n' \
    "${CONTAINER_NAME}" "${TP_SIZE}" "${NNODES}" "${NODE_RANK}" "${SPEC_TOKENS}" \
    "${KV_CACHE_DTYPE}" "${QUANTIZATION}" "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" \
    "${GPU_MEMORY_UTILIZATION}" "${SERVED_MODEL_NAME}" "${PORT}"

NETWORK_ARGS=(--ipc=host --shm-size 32g)
if [ "${NNODES}" -gt 1 ]; then
    NETWORK_ARGS+=(--network host)
else
    NETWORK_ARGS+=(-p "${PORT}:${PORT}")
fi

docker run \
    --name "${CONTAINER_NAME}" \
    --detach \
    --restart "${RESTART_POLICY}" \
    --init \
    --gpus all \
    "${NETWORK_ARGS[@]}" \
    --env HF_HUB_OFFLINE=1 \
    --env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
    --env NCCL_DEBUG="${NCCL_DEBUG}" \
    --env NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
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
    "${EAGER_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    "${QUANT_ARGS[@]}" \
    "${DIST_ARGS[@]}" \
    "${SPEC_ARGS[@]}"

printf '[%s] detached (%s); first boot compiles kernels. Follow: docker logs -f %s\n' \
    "${CONTAINER_NAME}" "${RESTART_POLICY}" "${CONTAINER_NAME}"
