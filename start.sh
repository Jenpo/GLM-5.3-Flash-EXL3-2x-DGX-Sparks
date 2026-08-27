#!/usr/bin/env bash
# ============================================================================
# start.sh — Spark runtime for GLM-5.3-Flash EXL3 (SM121 / GB10)
# ============================================================================
#
# We serve brandonmusic/GLM-5.3-Flash-EXL3-4bpw on this 2× DGX Spark (GB10 /
# SM121) kit: vLLM TP=2 over CX7, OpenAI API on :8888, NoPE-MLA overlay image.
#
#   head   : this machine (HEAD_IP, default 10.0.0.1) — vLLM rank 0 + API
#   worker : WORKER_USER@WORKER_IP (default: $USER@10.0.0.2) — vLLM rank 1, --headless
#   layout : --tensor-parallel-size 2, --nnodes 2, mp executor (not Ray)
#
# EXL3, not NVFP4. Do not pass --moe-backend marlin.
#
# What we do:
#   1. preflight  — docker/ssh/disk on both nodes
#   2. image      — docker pull IMAGE from GHCR if missing; ship to the
#                   worker with docker save | ssh docker load. BUILD=1
#                   rebuilds the overlay from this repo instead.
#   3. download   — EXL3 weights into the local HF cache if missing (~164 GiB)
#   4. sync       — rsync that cache to the worker (each rank loads local disk)
#   5. launch     — worker --headless, then head + `vllm serve` (both
#                   --network host --ipc=host)
#   6. wait       — poll /health up to READY_TIMEOUT
#
# Usage:
#   ./start.sh                    start (download/sync/launch) — default
#   ./start.sh stop               stop both nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers + API health
#   ./start.sh logs               follow head logs
#   ./start.sh logs worker        follow worker container logs
#
# Node IPs live in .env (copied from .env.example on first run).
# Handy overrides: SKIP_DOWNLOAD=1 SKIP_SYNC=1 PULL=1 BUILD=1 TAIL=1 HF_TOKEN=...
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || {
        echo "ERROR: missing .env.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[glm53-exl3]\033[0m wrote .env from .env.example — edit HEAD_IP / WORKER_IP if needed\n'
fi
# Caller exports (MTP_TOKENS=2 ./start.sh restart) must win over .env.
_cli_mtp="${MTP_TOKENS-}"
_cli_eager="${ENFORCE_EAGER-}"
_cli_fused="${EXL3_FUSED_MOE-}"
_cli_image="${IMAGE-}"
_cli_util="${GPU_MEM_UTIL-}"
_cli_lm="${LANGUAGE_MODEL_ONLY-}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
[ -n "${_cli_mtp}" ] && MTP_TOKENS="$_cli_mtp"
[ -n "${_cli_eager}" ] && ENFORCE_EAGER="$_cli_eager"
[ -n "${_cli_fused}" ] && EXL3_FUSED_MOE="$_cli_fused"
[ -n "${_cli_image}" ] && IMAGE="$_cli_image"
[ -n "${_cli_util}" ] && GPU_MEM_UTIL="$_cli_util"
[ -n "${_cli_lm}" ] && LANGUAGE_MODEL_ONLY="$_cli_lm"

# ----------------------------- configuration -------------------------------
MODEL="${MODEL:-brandonmusic/GLM-5.3-Flash-EXL3-4bpw}"
MODEL_CACHE_NAME="${MODEL_CACHE_NAME:-models--${MODEL//\//--}}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
GHCR_USER="${GHCR_USER:-MiaAI-Lab}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
# Same OS user on both Sparks unless .env sets WORKER_USER (mixed-account kits).
WORKER_USER="${WORKER_USER:-$USER}"
if [ "$WORKER_USER" = "$USER" ]; then
    WORKER_HOME="${WORKER_HOME:-$HOME}"
else
    WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
fi
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
# glm53-flash already ships nvidia-nccl. LD_PRELOAD of the host 2.30.7 SO
# makes DeepEP assert duplicate NCCL (/nccl/... vs nvidia/nccl/lib/...).
# Set USE_HOST_NCCL=1 only if image NCCL cannot talk CX7.
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"

TP="${TP:-2}"
NNODES="${NNODES:-2}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29521}"

MTP_TOKENS="${MTP_TOKENS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
LIMIT_MM="${LIMIT_MM:-{\"image\":4,\"video\":1}}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
# Python EXL3 expert loop is not CUDA-graph friendly on this UMA; graphs were
# ~+2% on prior GB10 GLM work anyway.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# 1 = fused exl3_moe (decode). 0 restores the unique-expert LinearEXL3 loop.
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"

CONTAINER_HEAD="${CONTAINER_HEAD:-glm53-exl3-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-glm53-exl3-worker}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-glm53-flash}"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-exl3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

worker_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

usage() { sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

count_shards() {
    find "$1/snapshots" -name '*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

ensure_refs_main() {
    local ref="$MODEL_PATH/refs/main" snap
    [ -f "$ref" ] && [ -n "$(<"$ref")" ] && return 0
    snap="$(ls -1t "$MODEL_PATH/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$snap" ] || die "no snapshots under $MODEL_PATH — re-run download"
    mkdir -p "$MODEL_PATH/refs"
    printf '%s' "$snap" >"$ref"
    log "wrote refs/main -> $snap (hf download left it empty)"
}

resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash dir
    ensure_refs_main
    hash="$(<"$ref")"
    dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    log "checking worker ${WORKER_SSH} ..."
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to ${WORKER_SSH} — set up passwordless ssh first"
    worker_ssh "docker info >/dev/null 2>&1" \
        || die "worker cannot talk to its docker daemon (docker group?)"
    worker_ssh "nvidia-smi -L 2>/dev/null | grep -q GB10" \
        || warn "no GB10 GPU visible on worker"

    [ "$TP" = "2" ] || warn "TP=${TP} on a 2×1-GPU cluster — expected TP=2"
    [ "$NNODES" = "2" ] || warn "NNODES=${NNODES} — expected 2"

    local others
    others=$(docker ps --format '  {{.Names}}  ({{.Image}})' | grep -v "^  ${CONTAINER_HEAD}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the head:"
        echo "$others" >&2
        warn "this model needs most of each GB10 — stop GPU containers first"
    fi
    others=$(worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${CONTAINER_WORKER}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the worker:"
        echo "$others" >&2
    fi

    check_port_free "$PORT" PORT
    check_port_free "$MASTER_PORT" MASTER_PORT

    local need_kb=$((180 * 1024 * 1024)) avail
    mkdir -p "$HF_CACHE_DIR"
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~164 GiB model"
    avail=$(worker_ssh "df -Pk '$WORKER_HOME' 2>/dev/null" | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker for a ~164 GiB model"

    log "preflight OK (head=$(hostname) ${HEAD_IP}, worker=${WORKER_SSH})"
}

# ------------------------------ image --------------------------------------
image_from_registry() {
    case "$IMAGE" in
        */*) return 0 ;;
        *) return 1 ;;
    esac
}

login_ghcr_if_token() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
}

build_image() {
    log "building ${IMAGE} from Dockerfile (log: $LOGDIR/build-sm121.log) ..."
    docker build -t "$IMAGE" "$SCRIPT_DIR" \
        >"$LOGDIR/build-sm121.log" 2>&1 \
        || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
}

pull_image() {
    login_ghcr_if_token
    log "pulling ${IMAGE} ..."
    docker pull "$IMAGE" && return 0
    die "docker pull ${IMAGE} failed (GHCR package is private).
  echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
  PAT needs read:packages. Or set GHCR_TOKEN + GHCR_USER in .env.
  Overlay rebuild instead: BUILD=1 ./start.sh"
}

ship_image_to_worker() {
    log "shipping ${IMAGE} to worker via docker save | ssh docker load ..."
    docker save "$IMAGE" | worker_ssh docker load >/dev/null
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_id="" worker_id="" refresh=0
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
    fi
    if worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        worker_id="$(worker_ssh "docker image inspect -f '{{.Id}}' '$IMAGE'")"
        if [ -n "$head_id" ] && [ "$worker_id" = "$head_id" ]; then
            worker_ok=1
        else
            worker_ok=0
            log "worker image id differs — will ship ${IMAGE}"
        fi
    fi
    if [ "${BUILD:-0}" = "1" ]; then
        build_image
        refresh=1
    elif [ "$head_ok" = "0" ] || [ "${PULL:-0}" = "1" ]; then
        if image_from_registry && [ "${SKIP_PULL:-0}" != "1" ]; then
            pull_image
        else
            build_image
        fi
        refresh=1
    fi
    if [ "$worker_ok" = "0" ] || [ "$refresh" = "1" ]; then
        ship_image_to_worker
    fi
    if [ "${SKIP_OVERLAY_VERIFY:-0}" != "1" ]; then
        log "GPU EXL3 self-check on ${IMAGE} (log: $LOGDIR/overlay-verify.log) ..."
        docker run --rm --gpus all \
            -e EXL3_SELFCHECK_GPU=1 \
            --entrypoint python3 "$IMAGE" /opt/glm53/test_exl3_overlay.py \
            >"$LOGDIR/overlay-verify.log" 2>&1 \
            || { tail -n 80 "$LOGDIR/overlay-verify.log" >&2; die "EXL3 overlay GPU self-check failed"; }
        log "overlay verify OK"
    fi
    log "image ready on both nodes"
}

# ---------------------------- weight download ------------------------------
download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    local need=0 have
    have="$(count_shards "$MODEL_PATH")"
    if [ ! -d "$MODEL_PATH" ]; then
        need=1
    elif [ "${have:-0}" -lt "$EXPECTED_SHARDS" ]; then
        need=1
        log "weights incomplete ($have / $EXPECTED_SHARDS shards) — resuming download"
    elif [ "${REFRESH_WEIGHTS:-0}" = "1" ]; then
        need=1
    fi
    [ "$need" = "0" ] && { log "weights already present: $MODEL_PATH ($have shards)"; ensure_refs_main; return; }

    local hf
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"

    mkdir -p "$HF_CACHE_DIR"
    log "downloading ${MODEL} (~164 GiB / ${EXPECTED_SHARDS} shards) into ${HF_CACHE_DIR} ..."
    HF_HOME="$HF_CACHE_DIR" "$hf" download "$MODEL"
    ensure_refs_main
    have="$(count_shards "$MODEL_PATH")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] \
        || die "download finished with $have / $EXPECTED_SHARDS shards"
    log "download complete ($have shards)"
}

# ------------------------------ weight sync --------------------------------
sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to worker"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    log "syncing weights to worker (first run moves ~164 GiB over the p2p link) ..."
    worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub'"
    rsync -a --partial --info=progress2 \
        "$MODEL_PATH/" "${WORKER_SSH}:${WORKER_CACHE_DIR}/hub/${MODEL_CACHE_NAME}/"
    log "worker weights in sync"
}

# ------------------------ inner container scripts --------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-head] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-worker] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 1
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --headless
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
say "joining TP2 at ${HEAD_IP}:${MASTER_PORT} as rank 1"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
launch_cluster() {
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true

    mkdir -p "$CACHE_ROOT"
    worker_ssh "mkdir -p '$WORKER_VLLM_CACHE'"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e HF_HOME=/root/.cache/huggingface
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT"
    )
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=() worker_preload=""
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
        if worker_ssh "test -f '$WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME'"; then
            worker_preload="-v '$WORKER_NCCL_HOST_DIR:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
            log "worker: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "worker: $WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local serve_env=""
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             KV_CACHE_DTYPE MTP_TOKENS LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             LIMIT_MM ENFORCE_EAGER EXL3_FUSED_MOE MODEL_DIR EXTRA_ARGS; do
        serve_env+=" -e $v='${!v:-}'"
    done

    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$WORKER_CACHE_DIR:/root/.cache/huggingface' \
        -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        ${worker_preload} \
        ${worker_nccl} \
        -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' \
        -e VLLM_HOST_IP='$WORKER_IP' \
        ${serve_env} \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null

    log "starting head (vLLM API :${PORT}; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$CACHE_ROOT:/root/.cache/vllm" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e PORT="$PORT" -e TP="$TP" -e NNODES="$NNODES" \
        -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
        -e QUANTIZATION="$QUANTIZATION" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e MTP_TOKENS="$MTP_TOKENS" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
        -e MODEL_DIR="$MODEL_DIR" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on a 320B MoE is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            log "head container exited during startup"
            exited=1; break
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "head container exited after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" >"$LOGDIR/worker.log" 2>&1 || true
}

on_ready() {
    log "======================================================================"
    log "GLM-5.3-Flash EXL3 is UP (TP=${TP}, nnodes=${NNODES})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN: ${HEAD_IP}:${PORT})"
    log "  model name : ${SERVED_MODEL_NAME}"
    log "  weights    : ${MODEL}  quant=${QUANTIZATION}  kv=${KV_CACHE_DTYPE}"
    local vision=on
    [ "${LANGUAGE_MODEL_ONLY}" = "1" ] && vision=off
    log "  features   : tools=glm47+auto, reasoning=glm45, MTP k=${MTP_TOKENS}, vision=${vision}"
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"hello!\"}]}'"
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker | ./start.sh stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — containers keep running"; exit 130' INT
    fi
}

# ------------------------------- start -------------------------------------
start() {
    preflight
    ensure_image
    download_weights
    sync_weights
    write_inner_scripts

    MODEL_DIR="$(resolve_model_dir)"
    log "model load path (in-container): ${MODEL_DIR}"
    log "config: image=${IMAGE} tp=${TP} nnodes=${NNODES} quant=${QUANTIZATION} mtp=${MTP_TOKENS} max-len=${MAX_MODEL_LEN} gpu-util=${GPU_MEM_UTIL} kv=${KV_CACHE_DTYPE} lm-only=${LANGUAGE_MODEL_ONLY} port=${PORT}"

    launch_cluster
    if wait_for_health; then
        on_ready
        return
    fi
    collect_failure_logs
    echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
    tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker log ($LOGDIR/worker.log) ----"
    tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
    log "stopped."
}

# ------------------------------ status -------------------------------------
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"
    worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
        || log "  (worker unreachable)"
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        worker)
            log "following worker container logs on ${WORKER_SSH} ..."
            trap '' INT
            worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

# ------------------------------- main --------------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start)   shift || true; start ;;
        stop)    stop ;;
        restart) stop; start ;;
        status)  status ;;
        logs)    shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
