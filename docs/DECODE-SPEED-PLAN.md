# GLM-5.3-Flash EXL3 decode speed — fused MoE + MTP k A/B

Plan for the two changes that can actually move tok/s on this 2× GB10 EXL3
serve. DFlash2 is out of scope: there is no GLM-5.3-Flash DFlash2 checkpoint.

Live API today: `http://127.0.0.1:8888/v1` (`glm53-exl3-head` /
`glm53-exl3-worker`, image `glm53-flash-sm121:local`). **Do not take it down
until Phase 0 receipts are on disk.**

## Done means

A healthy OpenAI API on `:8888` still serving
`brandonmusic/GLM-5.3-Flash-EXL3-4bpw` (EXL3 trellis / MCG, routed experts
only, TP=2 over CX7) that:

1. Runs routed experts through **one fused `exl3_moe` launch per layer** on
   the decode path (not the Python unique-expert `LinearEXL3` loop).
2. Has a measured **MTP k ∈ {0,1,2,5}** table on that fused path, with the
   default set to the winner (or k=5 if they are within 10%).
3. Has tried **CUDA graphs** (`ENFORCE_EAGER=0`) after the fused kernel is
   live; keeps graphs only if they do not regress tok/s or corrupt output.
4. Still boots `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla` + `EXL3 MCG
   engaged`, coherent completions, glm47 tools + glm45 reasoning.

Ship: overlay + tests + `.env`/`start.sh` defaults that match the winner.
Do not leave the cluster on a losing flag.

## Hard facts (do not violate)

- Checkpoint: `brandonmusic/GLM-5.3-Flash-EXL3-4bpw` (~164 GiB, 120 shards,
  `quant_method=exl3`, `codebook=mcg`, `scope=glm53_routed_experts_only`).
  **Do not destroy HF weights. Do not requantize.**
- Image family: `glm53-flash-sm121:local` FROM
  `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…`. Arm64, SM121
  cubins (`TORCH_CUDA_ARCH_LIST=12.1a`). No qemu, no amd64, no
  `cstechdev/vllm:glm53-flash-nope-sm120-*`, no silent switch to NVFP4 /
  `glm53-flash-sm121:v8`.
- **Never** `--moe-backend marlin`. That is the NVFP4 path.
- Registering `"exl3"` is not enough: experts must stay packed
  (trellis + suh + svh + mcg) and run Trellis/MCG kernels.
- Cluster: spark1 head `10.0.0.1` + spark2 worker `zurih@10.0.0.2`, CX7
  (`enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`), TP=2, mp
  executor, `--nnodes 2`. `USE_HOST_NCCL=0`.
- Attention must stay `FLASHINFER_MLA_SPARSE_SM120` (NoPE pad into GLM_NSA).
  Abort if it becomes SM90 / SM100-only / TRITON_MLA-only.
- `--kv-cache-dtype fp8` (packed `fp8_ds_mla`). bf16 KV has no sparse kernel
  on this arch.
- Scratch only under the goal implementer dir. Do not force-push. Unwedge
  (`./start.sh stop`) on failure; restore the last healthy image+flags.
- Shared experts, attention, embeddings, lm_head stay native dtype.

## Why it is slow today

Measured on this kit (Python EXL3 loop, `ENFORCE_EAGER=1`, `MTP_TOKENS=5`):

| metric | value |
|---|---|
| median decode | ~9 tok/s (64-token structural) |
| NVFP4 sibling on same cluster | ~22 tok/s (Marlin, MTP-4) |
| DS4F EXL3 SparkInfer (1 Spark) | 44–47 tok/s |
| Qwen3.8-27B EXL3 + DFlash2 (1 Spark, dense) | 47.5 tok/s |

Bottleneck is `Exl3MoEMethod.apply` in `overlay/exl3.py`: for each unique
expert, three `LinearEXL3.forward` launches (gate / up / down) from Python.
GLM-5.3-Flash routes **8 of 288** experts per token. MTP-5 then asks for
**six** decode positions per step, and on SM12x:

- fused multi-step MTP draft is unsupported on `FLASHINFER_MLA_SPARSE_SM120`
- DeepGEMM fp8 paged MQA only serves indexer `next_n ∈ {1, 2}`
- k=5 therefore uses the flattening fallback (`start.sh` already documents this)

So k=5 is not “MTP-4 fused”. It is sequential/flattened speculation on top of
an already Python-bound MoE.

Reference fused kernel (already on this machine, already SM121-capable):

- `/home/mia/MiaAI_Lab/exl3/exllamav3/exllamav3_ext/quant/exl3_moe.cu`
- Python call site: `exllamav3/modules/block_sparse_mlp.py` (`ext.exl3_moe(...)`)
- Binding: `exllamav3_ext.exl3_moe` / `exl3_moe_max_concurrency`
- Pointer tables: `modules/multilinear.py` (`ptrs_trellis/suh/svh`)
- Decode-small-bsz graph path: `run_bszN` (mgemm, no host argsort)

The Docker image currently installs **upstream**
`EXLLAMAV3_COMMIT=c5d9c657` (0.0.43) which exposes `LinearEXL3` but is
**older than** the fused MoE sources in the local tree. Bump or vendor;
do not assume `import exllamav3_ext; ext.exl3_moe` exists in the live image.

Geometry we must hit (from boot logs):

- bits = 4, codebook = mcg (`gate_mcg/up_mcg/down_mcg = True`, mul1 = False)
- hidden = 4096, intermediate_local = 1024 (TP=2 column/row shard)
- silu+mul with `swiglu_limit` default 10.0
- w13 stacked `[E, 2, …]` (gate=0, up=1), w2 `[E, …]`
- trellis tile 16; K_gate = K_up = K_down = 4

`exl3_moe` launch shape:

- `hidden_state` fp16 `(T, 4096)`, `output_state` fp32 zeroed `(T, 4096)`
- `expert_count` int64 `(E+1,)`, `token_sorted` / `weight_sorted` grouped by expert
- temp buffers `(concurrency, TEMP_ROWS_FUSED=128, {hidden, intermediate})`
- `num_active` = experts with `0 < count <= 128`
- experts above 128 tokens fall back to the existing per-expert `LinearEXL3`
  loop (prefill). Decode (T small) must stay on the fused launch.

## Phases

Do not stack kernel + graphs + MTP change in one boot. Each phase after 0
stops both ranks, relaunches, benches, then either keeps or reverts.

Shared launch: `SKIP_DOWNLOAD=1 SKIP_SYNC=1` unless the image digest changed
(then ship with `PULL=1`). `READY_TIMEOUT=1800`. Capture `docker logs` on
head **and** worker **before** every stop.

### Phase 0 — live baseline (no restart)

Bench the current server as-is (`MTP_TOKENS=5`, `ENFORCE_EAGER=1`, Python
loop). If `/health` is down, boot production flags once, then bench.

Receipts (scratch + `logs/decode-speed-YYYYMMDD/`):

- `health` 200
- head log lines: `FLASHINFER_MLA_SPARSE_SM120`, `fp8_ds_mla`,
  `EXL3 MCG trellis engaged`, `num_speculative_tokens`, `enforce-eager`,
  KV pool tokens
- 3-run streaming decode (script below)
- one coherence check: temp=0 “capital of France” → Paris
- MTP accept stats if logged (mean accepted length)

Fill the results table. This is the only number later phases may beat.

### Phase 2a — MTP k A/B on the **current** Python loop

Flag-only. No image rebuild. Same weights, same eager, same MoE.

Order: **k=2 → k=1 → k=0**. Keep k=5 as Phase 0; do not re-bench it unless
the first k=2 boot looks broken.

| run | command | what we learn |
|---|---|---|
| 2a-k2 | `MTP_TOKENS=2 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart` | DeepGEMM `next_n=2` native indexer |
| 2a-k1 | `MTP_TOKENS=1 …` | cheapest spec, `next_n=1` |
| 2a-k0 | `MTP_TOKENS=0 …` | pure AR; lower bound |

Abort a k if: health fail, NaN/garbage, attention backend change, tok/s
worse than Phase 0 by >15% **and** TTFT not meaningfully better.

**Keep the fastest coherent k as `MTP_TOKENS` going into Phase 1.** If they
are within 10%, keep k=5 (native head width) so fused-MoE work is not blamed
on an MTP change.

Log grep must show whether the indexer used native `next_n` or flattening.

### Phase 1 — fused `exl3_moe` (the actual speed pass)

#### 1.0 Kernel into the image

Prefer compiling a **pinned** ExLlamaV3 that already has `exl3_moe` and
still builds on aarch64:

- Source of truth for the kernel: `/home/mia/MiaAI_Lab/exl3` (upstream
  `turboderp-org/exllamav3` + local aarch64 stubs). Record the git SHA in
  the Dockerfile `ARG`.
- Keep `overlay/patch_exl3_ext_aarch64.py` (AVX2/AVX512 CPU allreduce stubs).
- Keep `CPATH=…/nvidia/cu13/include` for `cusparse.h`.
- `TORCH_CUDA_ARCH_LIST=12.1a` only. Fail the build if `cuobjdump -lelf`
  on `exllamav3_ext` lacks `sm_121a`.
- After install, `hasattr(exllamav3_ext, "exl3_moe")` must be true.

If bumping the whole tree breaks the LinearEXL3 ABI or aarch64 compile, vendor
**only** `exl3_moe*.{cu,cuh}` + instances into the existing c5d9c657 tree and
add the `m.def("exl3_moe", …)` binding. Do not copy SparkInfer/b12x; that
stack is a different runtime.

Do not compile at docker-build with GPUs (`EXL3_SELFCHECK_GPU=0`). GPU
self-check stays post-build: `docker run --gpus all`.

#### 1.1 Overlay: pointer tables + fused apply

Edit `overlay/exl3.py` only (plus tests). `create_weights` / `_load_exl3` /
TP shard rules stay.

In `process_weights_after_loading`:

- Keep MCG marker check.
- Keep `_exl3_inners` (fallback for fat experts / kernel miss).
- Build device int64 pointer tensors, shape `(E,)`:

  `gate/up`: `w13_*[e, 0/1].data_ptr()`  
  `down`: `w2_*[e].data_ptr()`

- Allocate fused temps once per layer (or a module-level cache keyed by
  `(hidden, intermediate_local, concurrency)`):

  `concurrency = exl3_moe_max_concurrency(device)`  
  `temp_state_{g,u}: (C, 128, hidden) fp16`  
  `temp_intermediate_{g,u}: (C, 128, intermediate_local) fp16`

- Log: `EXL3 MCG trellis engaged … fused_moe=exl3_moe concurrency=N`.

In `apply`:

1. Flatten `topk_ids` / `topk_weights`; map through `layer.expert_map` the
   same way the Python loop does (skip `-1`).
2. `token_sorted`, `weight_sorted` via argsort of local expert ids;
   `expert_count = bincount(..., minlength=E+1)`.
3. `out = zeros(T, hidden, dtype=fp32)`.
4. `exllamav3_ext.exl3_moe(x.half(), out, expert_count, token_sorted,
   weight_sorted, temps…, act_fn=silu_mul, K=4,4,4, ptrs…, mcg=True,
   mul1=False, act_limit, num_active)`.
5. If any expert has `count > 128`, run **only those** through the existing
   LinearEXL3 loop (do not double-apply fused experts).
6. Return `out.to(x.dtype)`. Shared experts remain the runner’s job.

Must not host-sync on the decode hot path more than `exl3_moe` itself
requires. If `expert_count.tolist()` is required for `num_active`, compute
`num_active` on GPU (`((count > 0) & (count <= 128)).sum()`) and pass that;
pass `-1` only if the kernel documents it as “max concurrency”.

#### 1.2 Tests (`tests/test_exl3_overlay.py`)

Keep registry + TP shard + `sm_121a` checks. Add:

- `hasattr(exllamav3_ext, "exl3_moe")`
- Tiny GPU GEMM: random trellis-shaped packed tensors (or a captured one-expert
  fixture) — fused `exl3_moe` vs existing `execute_exl3_linear` loop, max
  abs err bound (fp16 noise, not bit-identical). Skip if `EXL3_SELFCHECK_GPU=0`.
- Apply-path unit: 2 tokens, 3 experts, top-k 2, expert_map with one `-1`.

Image-build still runs `EXL3_SELFCHECK_GPU=0`. Post-build GPU check is
mandatory before shipping to the worker.

#### 1.3 Serve

`PULL=1 SKIP_DOWNLOAD=1 ./start.sh restart` so head and worker image Ids
match (start.sh already compares Ids). Boot asserts:

- `fused_moe=exl3_moe` (new log line)
- no `Unknown quantization method: exl3`
- no Python-loop-only path on decode (optional nvtx / counter: fused launches
  per step > 0)
- coherence: Paris; `9.9 > 9.11`; one-sentence sky-blue
- bench vs Phase 0 / 2a

**Success bar:** fused+chosen-k decode median **≥ 1.5×** Phase 0, no
coherence regression. If fused is correct but <1.2×, still keep it (graphs
in 1.4 may be the rest) unless it is slower than the Python loop.

Fallback: env `EXL3_FUSED_MOE=0` restores the Python loop without rebuild.

### Phase 1.4 — CUDA graphs (only after fused is the default)

`ENFORCE_EAGER=0`. Capture sizes: `seqs × (k+1)`. With `MAX_NUM_SEQS=4`
and k=2 that is 12; start with `6,12,24` (DS4F Spark recipe) or
`cudagraph_capture_sizes` equivalent on this vLLM.

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` if this tree honors it (grows
KV; DS4F used this).

Abort graphs on: capture OOM, `Workspace is locked`, NaN, or tok/s worse
than fused+eager. Default back to `ENFORCE_EAGER=1`.

The fused kernel’s host argsort/bincount may **prevent** capture. If so,
document it and stop — do not invent a second graph path in the same night
unless `run_bszN`-style mgemm is a small overlay (decode T ≤ 8). That is a
stretch goal, not the ship gate.

### Phase 2b — MTP k A/B on the **fused** path

Repeat k=2, k=1, k=0, k=5 (and the Phase-2a winner) on the fused image.
Same bench. Set `.env` `MTP_TOKENS` to the winner. Update README.

Hypothesis: fused MoE makes extra verify tokens cheaper, so k=2 (native
indexer) may beat both k=5-flattened and k=0. Confirm; do not assume.

## Bench protocol (every phase)

Prompt (thinking off, temp 0, 200 tokens, 3 streaming runs, median):

> Write a detailed step-by-step explanation of how a hash map works,
> including collision handling, resizing, and time complexity. Be thorough.

Model id: `brandonmusic/GLM-5.3-Flash-EXL3-4bpw`.
`chat_template_kwargs.enable_thinking = false`.
`stream_options.include_usage = true`.

Decode tok/s = `(completion_tokens - 1) / (end - first_token_time)`.
Also record TTFT, `finish_reason`, NaN/`locklock`, completion_tokens.

Coherence (once per boot, non-stream ok): Paris; 9.9 vs 9.11.

Write results into the table below and into
`logs/decode-speed-YYYYMMDD/phase-*.json`.

## Results

| Phase | Config | Health | Backend | MoE | Eager | MTP k | KV tokens | TTFT med (s) | tok/s med | Coherent | Keep? | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| 0 | prod Python EXL3 | 200 | FLASHINFER_MLA_SPARSE_SM120 | loop | 1 | 5 | 1,627,477 | 1.41 | 10.03 | yes (Paris) | baseline | flattening next_n=6; glm47+glm45; no NaN; 3-run tok/s 9.99/10.04/10.03 |
| 2a-k2 | flags only | 200 | FLASHINFER_MLA_SPARSE_SM120 | loop | 1 | 2 | 1,714,482 | 1.21 | 12.32 | yes | carry | flattening still (next_n=3); +23% vs k=5 |
| 2a-k1 | flags only | 200 | FLASHINFER_MLA_SPARSE_SM120 | loop | 1 | 1 | 1,810,041 | 1.10 | 11.69 | yes | no | native indexer next_n=2 flattening=False; slower than k=2 |
| 2a-k0 | flags only | 200 | FLASHINFER_MLA_SPARSE_SM120 | loop | 1 | 0 | 2,299,717 | 1.09 | 8.43 | yes | no | pure AR next_n=1 flattening=False; slowest |
| 1 | fused exl3_moe | 200 | FLASHINFER_MLA_SPARSE_SM120 | **exl3_moe** | 1 | 2 | 1,988,678 | 0.45 | 23.74 | yes | yes | 2.37× Phase 0; 1.93× loop-k2; concurrency=6; Paris/9.9>9.11/sky-blue |
| 1.4 | fused + graphs | fail | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 0 | 2 | — | — | — | n/a | no | capture: CPU↔CUDA copy (argsort/bincount); abort to ENFORCE_EAGER=1 |
| 1.4b | graph-safe apply, sizes 6 12 24 | 200 | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 0 | 2 | ~1.97M | 0.54 | 19.95 | yes | no | capture OK; padded MTP 3→6; −16% vs eager |
| 1.4c | graph-safe apply, sizes 1 2 3 4 6 8 12 | 200 | FLASHINFER_MLA_SPARSE_SM120 | **exl3_moe** | 0 | 2 | ~1.97M | 0.45 | **24.55** | yes | **winner** | +3.4% vs fused eager; overlay pin+scatter_add_+num_active=-1 |
| 2b-k5 | fused | 200 | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 1 | 5 | 1,564,672 | 0.56 | 19.83 | yes | no | flattening next_n=6; slower than fused k=2 |
| 2b-k2 | fused | 200 | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 1 | 2 | 1,988,678 | 0.45 | 23.74 | yes | **winner** | same as Phase 1; 2.37× Phase 0 |
| 2b-k1 | fused | 200 | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 1 | 1 | 1,785,075 | 0.45 | 20.65 | yes | no | native next_n=2; slower than fused k=2 |
| 2b-k0 | fused | 200 | FLASHINFER_MLA_SPARSE_SM120 | exl3_moe | 1 | 0 | 2,326,528 | 0.36 | 12.36 | mixed | no | native next_n=1; faster than loop AR (8.43) but slowest fused |
| R | restored winner | 200×2 | FLASHINFER_MLA_SPARSE_SM120 | **exl3_moe** | 1 | 2 | 1,768,718 | 1.92 | 26.39 (1-run) | yes (Paris; 9.9>9.11; sky-blue) | **yes** | current boot: fused_moe=exl3_moe + fp8_ds_mla + EXL3 MCG; glm47+glm45; logs/decode-speed-20260827/winner-{head,worker}.log + winner-coherence.json |

## Restore / unwedge

```bash
cd /home/mia/NewModels/glm-5.3-flash-sm120
./start.sh stop
# last healthy: IMAGE=glm53-flash-sm121:local MTP_TOKENS=5 ENFORCE_EAGER=1
SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh
```

If a fused image is bad, retag/rebuild from the pre-Phase-1 Dockerfile
(commit the fused work on a branch; `main` launcher must still boot the
Python loop via `EXL3_FUSED_MOE=0` until 1.3 passes).

Never `docker rm` HF caches. Never `REFRESH_WEIGHTS=1`. Never force-push.

## Out of scope

- DFlash2 / DSpark drafters (no GLM-5.3-Flash draft weights)
- NVFP4 / Marlin / CUTLASS MoE
- Switching runtime to TabbyAPI / raw exllamav3 serve
- SparkInfer/b12x image
- Vision (`LANGUAGE_MODEL_ONLY` stays 1)
- Changing TP, CX7 pins, or NCCL
