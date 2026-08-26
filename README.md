# glm-5.3-flash-sm120

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on consumer
Blackwell (SM120). The stock `vllm/vllm-openai:glm53-flash` image loads the weights and
then dies on the first forward with `pe_dim must be 64 for fp8_ds_mla`, because this
checkpoint is NoPE MLA and SM120's only sparse-MLA kernel expects a 64-wide RoPE block.
This is a single-layer overlay on that image which makes it run.

Image:

```
cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
```

Verified on 4x RTX PRO 6000 Blackwell (96 GB, capability 12.0), CUDA 13.0.

## Run

```bash
docker run -d --name vllm-glm53-flash \
  --restart unless-stopped --init --gpus all --ipc=host --shm-size 32g \
  -p 8001:8001 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v "$MODEL_DIR":/model:ro \
  -v "$HOME/.cache/vllm-glm53-flash":/root/.cache \
  cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1 \
  /model \
  --served-model-name GLM-5.3-Flash \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 4 \
  --max-num-seqs 10 \
  --max-model-len 524288 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

`MODEL_DIR` is a local checkout of the checkpoint (62 shards, ~306 GiB); add
`-e HF_HUB_OFFLINE=1` when serving from it. To pull from the Hub instead, drop the
`-v "$MODEL_DIR"` line, pass `zai-org/GLM-5.3-Flash` in place of `/model`, and mount
`-v "$HOME/.cache/huggingface":/root/.cache/huggingface`.

`serve.sh` is the same command with env-overridable knobs (`MAX_MODEL_LEN`, `SPEC_TOKENS`,
`PORT`, ...).

First boot takes ~72 s to load weights plus ~200 s of torch.compile and Triton JIT; the
`/root/.cache` mount keeps later boots short.

## Notes

- `--kv-cache-dtype fp8` is mandatory. The SM120 sparse-MLA path only accepts the packed
  `fp8_ds_mla` layout; bf16 KV has no kernel on this arch.
- `--max-num-seqs 10` pairs with 5 speculative tokens: decode batches over 64 tokens fall
  off FlashInfer's specialized decode kernel, and `10 x (5 + 1) = 60`.
- Context costs ~8.7 KiB/token of KV. 1M context does not fit alongside
  `num_speculative_tokens: 5` — use 1 or 0 for that.
- The command above is the verified configuration: 609,172 tokens of KV (1.16x concurrency
  at full context), engine init 185 s, MTP mean acceptance length 2.5-5.1 depending on
  workload (avg draft acceptance 29-82%).

## Build

```bash
docker build -t glm53-flash-sm120:local .
```

The `Dockerfile` pins the base by digest and documents each change inline.
