<h1 align="center">GLM-5.3 Flash EXL3 for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

OpenAI-compatible vLLM serve of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as
**[brandonmusic/GLM-5.3-Flash-EXL3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw)**
(routed-experts-only EXL3/MCG trellis, 4 bpw, ~164 GiB, 120 shards) on a **2× NVIDIA GB10**
kit: tensor-parallel size 2 over CX7, native `sm_121a` cubins, API on `:8888`.

This is **EXL3 weights + fp8 KV**, not NVFP4. Do not pass `--moe-backend marlin`.
Do not use the amd64 SM120 image `cstechdev/vllm:glm53-flash-nope-sm120-*`.

Measured on this kit (fused MoE, MTP k=2, enforce-eager): **~24 tok/s decode**
(3-run median) vs **~10 tok/s** on the old unique-expert Python loop at MTP k=5.
KV pool **2,051,954 tokens** at util **0.87** language-only; default serve is util **0.875** with vision (image+video).

## What runs

| Layer | Runtime |
|---|---|
| API | vLLM OpenAI (`/v1/chat/completions`) on the head, port **8888** |
| Model id | `brandonmusic/GLM-5.3-Flash-EXL3-4bpw` |
| Image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` FROM `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (arm64, CUDA 13.0) |
| Executor | `mp`, `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | this machine, `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (this kit: `zurih@10.0.0.2`), `--headless`, `glm53-exl3-worker` |
| Fabric | CX7 QSFP: `enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`. Image NCCL (`USE_HOST_NCCL=0`) |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA padded into GLM_NSA 576-wide) |
| KV | `--kv-cache-dtype fp8` → packed **`fp8_ds_mla`**, 656 B/token MLA record |
| Experts | packed trellis + suh + svh + mcg, codebook MCG, **one fused `exllamav3_ext.exl3_moe` launch per layer** |
| Dense / shared / attn / embed / lm_head | native (unquantized) |
| Spec | MTP, default **`MTP_TOKENS=2`** (measured winner) |
| Tools / reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | off (`ENFORCE_EAGER=1`) — capture hits a CPU↔CUDA copy in fused apply |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) — image + video, `--limit-mm-per-prompt {image:4,video:1}`, `--skip-mm-profiling` |

Kernels: `TORCH_CUDA_ARCH_LIST=12.1a`. ExLlamaV3 pin `c5d9c657` (0.0.43) exposes
`exl3_moe` / `exl3_moe_max_concurrency`; aarch64 CPU allreduce stubs in
`overlay/patch_exl3_ext_aarch64.py`.

## Why the overlay exists

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads this checkpoint and dies on
the first forward: `pe_dim must be 64 for fp8_ds_mla`. GLM-5.3-Flash is **NoPE MLA**
(`qk_rope_head_dim=0`, `kv_lora_rank=512`). On SM12x the only sparse-MLA backend is
`FLASHINFER_MLA_SPARSE_SM120`, whose packed record is 512 NoPE + 16 B scales + 128 B
RoPE (656 B). The overlay zero-pads the 512-d latent into that GLM_NSA geometry
(RoPE pad is zeros; the QK dot is unchanged) and registers a real EXL3 method so
routed experts stay packed instead of expanding to BF16.

Registering the name `"exl3"` is not enough. Experts must stay **trellis + suh +
svh + mcg** and run Trellis/MCG. Shared experts, attention, embeddings, and
`lm_head` stay native. TP=2 shards gate/up **column-wise** and down **row-wise**;
the MoE runner all-reduces once per layer.

## KV cache

`--kv-cache-dtype fp8` is required. The SM12x sparse-MLA kernel only accepts packed
`fp8_ds_mla`. **bf16 KV has no sparse kernel** on this arch.

| MTP k | MoE | KV tokens (util 0.85, this kit) | Indexer |
|---:|---|---:|---|
| 5 | fused | 1,564,672 | flattening, `next_n=6` |
| **2 (default)** | **fused** | **2,051,954** at util **0.87** (1,768,718 at 0.85) | flattening, `next_n=3` |
| 1 | fused | 1,785,075 | native DeepGEMM, `next_n=2` |
| 0 | fused | **2,326,528** | native, `next_n=1` |

MLA KV is **~8.7 KiB/token** once indexer + KDA page alignment are included
(656 B record is only the MLA slab). Language-only at **`GPU_MEM_UTIL=0.87`**
allocated **19.77 GiB** KV (2,051,954 tokens). Default is **0.875 + vision**:
the tower is **1.05 GiB BF16** (replicated per rank). Vision at 0.85 was
**12.63 GiB / 1,494,049 tokens**. That still covers a native 1M request. Weights + non-torch ~83 GiB
of 121 GiB UMA; vision adds ~1 GiB. Keep **`SKIP_MM_PROFILING=1`** — a max-size
image+video dummy profile OOMs this UMA. `LIMIT_MM={"image":4,"video":1}`.

**NVFP4 KV is not available here.** `FLASHINFER_MLA_SPARSE_SM120` only lists
`auto` / `fp8` / `fp8_e4m3` / `fp8_ds_mla`. FlashInfer’s SM12x NVFP4 attention
kernels are **dense MHA** (XQA/FA2), not sparse MLA. Adding `nvfp4` to the Python
dtype list will fail backend selection or `concat_and_cache_mla`. A real NVFP4
sparse-MLA cache would need new SM121 kernels plus a new packed write path — not
an `.env` flag. Do not confuse that with NVFP4 **weights** (`--moe-backend marlin`).

## Decode speed (this kit)

Protocol: thinking off, temp 0, 200 tokens, 3 streaming runs, median.
Prompt: hash-map explanation (see `tests/bench_decode.py`). Decode tok/s =
`(completion_tokens - 1) / (end - first_token_time)`.

| MoE | MTP k | tok/s med | TTFT med (s) | Keep |
|---|---:|---:|---:|---|
| Python unique-expert loop | 5 | 10.03 | 1.41 | baseline |
| loop | 2 | 12.32 | 1.21 | |
| loop | 1 | 11.69 | 1.10 | native indexer |
| loop | 0 | 8.43 | 1.09 | AR floor |
| **fused `exl3_moe`** | **2** | **23.74** | **0.45** | **default** |
| fused | 5 | 19.83 | 0.56 | flattening |
| fused | 1 | 20.65 | 0.45 | native indexer |
| fused | 0 | 12.36 | 0.36 | AR |

Fused k=2 is **2.37×** the loop-k=5 baseline and faster than the loop at every k.
`EXL3_FUSED_MOE=0` restores the Python loop without a rebuild (~12 tok/s at k=2).

CUDA graphs (`ENFORCE_EAGER=0`) fail at capture:
`Cannot copy between CPU and CUDA tensors during CUDA graph capture` (host
`argsort`/`bincount` on the fused apply path). Stay on `ENFORCE_EAGER=1`.

Full receipts: [`docs/DECODE-SPEED-PLAN.md`](docs/DECODE-SPEED-PLAN.md) and
`logs/decode-speed-YYYYMMDD/`.

Coherence on the winner boot (temp 0): capital of France → **Paris**;
**9.9 > 9.11**; a one-sentence sky-blue line. glm47 tools + glm45 reasoning stay
on the launcher.

## Quick start (2× Spark)

```bash
# GHCR is private — once per machine (PAT with read:packages)
echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin

cp .env.example .env          # edit HEAD_IP / WORKER_IP / WORKER_USER if needed
./start.sh                    # pull image, download EXL3, rsync, launch TP=2
```

First run of `./start.sh` copies `.env.example` → `.env` if missing. Prefix env
wins over `.env` (`MTP_TOKENS=1 SKIP_DOWNLOAD=1 ./start.sh restart`).

`./start.sh` will:

1. Preflight docker/ssh/disk on both nodes
2. `docker pull` `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` if that tag is missing on the head (or whenever `PULL=1`); then `docker save | ssh docker load` onto the worker
3. Download the EXL3 repo into `$HF_HOME` / `~/.cache/huggingface` (~164 GiB, 120 shards)
4. `rsync` that cache to `${WORKER_HOME}/.cache/huggingface`
5. Start rank 1 `--headless` on the worker, rank 0 + API on the head
6. Poll `/health` (weight load + warmup is slow; `READY_TIMEOUT` default 3600s)

The worker does not need GHCR credentials — only the head pulls, then ships the image over SSH.

```bash
SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh     # weights already local on both nodes
PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart   # re-pull GHCR + ship
BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # rebuild overlay from this repo + ship
./start.sh status
./start.sh logs                # head
./start.sh logs worker
./start.sh stop                # or ./stop.sh
```

Do not pull `glm53-flash-sm121:v8` — that is the older NVFP4/Ray kernel.

API: `http://127.0.0.1:8888/v1` (LAN: `http://10.0.0.1:8888/v1`).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "brandonmusic/GLM-5.3-Flash-EXL3-4bpw",
    "messages": [{"role": "user", "content": "hello!"}]
  }'
```

Needs: Docker (no sudo) on both nodes, passwordless SSH head → worker,
`docker login ghcr.io` on the head (private image), `hf` / `huggingface-cli` +
`curl` + `rsync` on the head, ~180 GiB free per node for the first download.
Mixed OS accounts: set `WORKER_USER` (this kit uses `zurih` on spark2).

NCCL cannot use the `10.0.0.x` loopback aliases — leave the CX7 pins unless
your cabling differs. `ncclCommInitRank` hangs without them.

## .env

| Knob | Default | What |
|---|---|---|
| `HEAD_IP` | `10.0.0.1` | this node, NCCL/vLLM master |
| `WORKER_IP` | `10.0.0.2` | other Spark |
| `WORKER_USER` | *(unset = `$USER`)* | SSH user on the worker |
| `WORKER_HOME` | `$HOME` if same user, else `/home/$WORKER_USER` | worker HF cache |
| `MODEL` | `brandonmusic/GLM-5.3-Flash-EXL3-4bpw` | Hub repo into the HF cache |
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | serve image (`PULL=1` re-pulls; `BUILD=1` rebuilds the overlay) |
| `GHCR_TOKEN` / `GHCR_USER` | *(unset)* | optional `docker login ghcr.io` before pull |
| `PORT` | `8888` | OpenAI API on the head |
| `TP` / `NNODES` | `2` / `2` | do not change for this recipe |
| `QUANTIZATION` | `exl3` | overlay method; never `marlin` |
| `MTP_TOKENS` | `2` | fused-path winner vs `{0,1,5}` |
| `ENFORCE_EAGER` | `1` | graphs fail capture on fused apply |
| `EXL3_FUSED_MOE` | `1` | `exl3_moe` per layer; `0` = LinearEXL3 loop |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not `nvfp4`, not bf16 |
| `GPU_MEM_UTIL` | `0.875` | GB10 UMA budget |
| `MAX_MODEL_LEN` | `1048576` | native `text_config.max_position_embeddings` |
| `MAX_NUM_SEQS` | `4` | decode batch; MTP adds k+1 tokens/seq |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | prefill chunk; 8192 oversubscribes GB10 indexer topk on long context |
| `LANGUAGE_MODEL_ONLY` | `0` | load vision tower (image + video) |
| `SKIP_MM_PROFILING` | `1` | skip max-size MM dummy at init (OOM otherwise) |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | NCCL sockets |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0` | NCCL HCAs |
| `USE_HOST_NCCL` | `0` | image nvidia-nccl; host preload duplicates DeepEP |

## Image / overlay

```bash
docker build -t glm53-flash-sm121:local .
# or: BUILD=1 ./start.sh
```

`./start.sh` **pulls** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` if
that tag is missing (`PULL=1` re-pulls). `BUILD=1` rebuilds the overlay from
this Dockerfile instead. After CUDA compile, Python overlay edits
(`overlay/exl3.py`, tests) are a cheap layer so they do not rebuild
`exllamav3_ext`.

| Path | Role |
|---|---|
| `Dockerfile` | NoPE sparse-MLA patches + EXL3 install (`sm_121a`) + self-check |
| `overlay/exl3.py` | `Exl3Config` / packed load / TP shard / fused `exl3_moe` apply |
| `overlay/patch_exl3_ext_aarch64.py` | stub AVX CPU allreduce so the ext builds on GB10 |
| `overlay/patch_model_overrides.py` | `"exl3"` in ModelConfig overrides |
| `tests/test_exl3_overlay.py` | registry, TP shard, `sm_121a` cubin, fused vs loop GEMM, `EXL3_FUSED_MOE=0` |
| `tests/bench_decode.py` | streaming decode + coherence probes against `:8888` |
| `start.sh` / `stop.sh` | 2-node launch |
| `files/chat_template.jinja` | GLM-5.3 MM template (`<|image|>` / `<|video|>`); checkpoint jinja is language-only |

Image-build runs `EXL3_SELFCHECK_GPU=0`. `./start.sh` runs the GPU self-check
(`docker run --gpus all`) before shipping unless `SKIP_OVERLAY_VERIFY=1`.

## Do not

- Destroy HF weights, requantize, `REFRESH_WEIGHTS=1`, or `docker rm` HF caches
- `--moe-backend marlin`, NVFP4 weights, or `glm53-flash-sm121:v8` as this serve
- qemu / amd64 / `cstechdev/vllm:glm53-flash-nope-sm120-*`
- `--kv-cache-dtype nvfp4` or bf16 (no sparse-MLA kernel)
- Change TP, CX7 pins, or `USE_HOST_NCCL` unless you are re-plumbing NCCL
- Force-push
