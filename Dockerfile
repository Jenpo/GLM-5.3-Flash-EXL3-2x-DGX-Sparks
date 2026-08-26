# GLM-5.3-Flash NoPE sparse MLA on SM120 (RTX PRO 6000 Blackwell).
#
# The stock image cannot serve this checkpoint on consumer Blackwell. GLM-5.3-Flash is
# NoPE MLA (qk_rope_head_dim=0, d_qk=512), and on SM120 the only sparse-MLA backend is
# FLASHINFER_MLA_SPARSE_SM120, whose impl mandates the packed fp8_ds_mla record. Two
# things then reject it:
#
#   1. concat_and_cache_mla (csrc/.../cache_kernels.cu:866) -> "pe_dim must be 64 for
#      fp8_ds_mla", because the 656 B record is 512 NoPE + 16 B scales + 128 B RoPE.
#   2. flashinfer/mla/_core.py:583 -> the SM120 sparse entry requires kv_lora_rank=512,
#      qk_rope_head_dim=64 and query head dim 576.
#
# FlashInfer instantiates three SM120 sparse-MLA model types
# (flashinfer/mla/_sparse_mla_sm120.py:125-170): DSV3_2 and GLM_NSA at d_qk=576 /
# 656 B with topk in {128,512,1024,2048}, and DSV4 at d_qke=512 / 584 B (448 fp8 +
# 64 bf16 + 8 B footer) with topk in {128,512,1024} only. GLM-5.3-Flash needs
# d_qk=512 with index_topk=2048, which no kernel instantiates, and its uniform 512-dim
# latent does not fit the DSv4 448/64 split.
#
# So run it in the GLM_NSA geometry the GLM-5.2 line already uses: zero-pad the 512-dim
# latent to 576. A zero RoPE block adds nothing to the QK dot, softmax is unchanged, and
# the value is taken from the 512 NoPE region (d_v=512), so the result is exact. It costs
# 656 B/token instead of ~528 (~24% more DSA KV over 11 DSA layers) and needs no kernel
# work. MLAAttentionSpec.real_page_size_bytes already allocates block_size * 656 for
# fp8_ds_mla on non-deepseek_v4 models, so allocation already matches.
#
# The impl also inherits MLAAttentionImpl.forward_mha, which raises NotImplementedError
# (only the SM100 sibling gets a real one via SparseMLACommonImpl). Declaring
# supports_dense_mha_prefill=False routes prefill through the same top-k MQA kernel --
# the same declaration rocm_aiter_mla_sparse.py:668 makes -- which is the model's own
# sparse semantics, and it also clears the builder's prefill metadata so use_dense_mha
# can never select the missing path.
#
# Third, the SM120 impl passes index_topk as both the page-table width and the top-k,
# which this checkpoint breaks: index_kpool=4 widens the indexer's top-k buffer past
# index_topk and rounds it up to a multiple of 128 (2048 -> 2176), so the kernel rejects
# the table ("expects sparse block_tables shape (256, 1, 2048), got (256, 1, 2176)").
# The SM100 sibling treats sparse_mla_top_k as page-table *capacity* and bounds each
# query by its own valid count (flashinfer_mla_sparse.py:461-476), so ask the index
# converter for those counts and pass the buffer width. Zero-length rows are pointed at
# slot 0 with length 1 and their output zeroed afterwards, mirroring the same file's
# empty-row handling.
#
# Capacity alone is not enough on SM120: the top-k width is a kernel template parameter,
# not a runtime bound. sparse_mla_sm120_prefill.cu:220 rejects any GLM_NSA/DSV3_2 prefill
# with topk != 2048, its dual-cache (two-segment) dispatcher is DSV4-only (line 406), and
# the DSV4 dual path hardcodes a 128-wide primary. _DECODE_DSV3_2_DISPATCH likewise only
# instantiates topk in {128,512,1024,2048}. So the candidate set has to be exactly 2048
# wide, while this model's indexer emits up to 2051: 512 pools expanded by index_kpool=4,
# plus an always-selected tail of up to kpool-1 tokens that have no compressed pool entry
# yet (glm5next/nvidia/model.py:606-616, ops/kpool_compress.py:950-967).
#
# Fourth, the indexer's DeepGEMM paged-MQA page size. deepgemm attention.hpp:320 accepts
# block_kv in {32,64,128} on sm_10x but only 64 on sm_120 (non-FP4 cache), while
# vllm/utils/deep_gemm.py hardcodes PAGED_MQA_PAGE_SIZES = (32, 64) arch-blind. The kpool
# indexer's storage block is block_size // index_kpool, and cuda.py aligns block_size to
# index_kpool * min(PAGED_MQA_PAGE_SIZES) = 128; the hybrid KDA page constraint then
# lands on block_size 1664, giving a 416-entry storage block. 416 is not a multiple of
# 64, so compressed_kernel_block_size (v1/worker/utils.py:69) falls back to a 32-entry
# page and DeepGEMM asserts. Aligning to index_kpool * 64 = 256 moves block_size to 1792,
# whose 448-entry storage block does tile by 64.
#
# That raise also has to not change the MLA kernel page: select_kernel_block_size walks
# the backend's int sizes in descending order, so [64, 256] would pick 256 for a 1792
# manager block, and every GLM_NSA/DSV3_2 kernel is instantiated at PAGE_BLOCK_SIZE=64
# only (_DECODE_DSV3_2_PAGE_BLOCK_SIZE, launch_prefill_sg<..., 2048, 64>). At block_size
# 1664 that was masked -- 256 does not divide 1664 -- so pin the SM120 backend to [64].
#
# Drop the lowest-ranked selected pool instead of the tail: expand 511 pools (2044
# tokens) + 3 tail = 2047 candidates in a 2048-wide buffer. That gives up 4 of 2048
# candidate tokens (0.2%), the *last* ones in top-k rank order, and keeps the recent
# tail, which carries the highest attention weight in a causal model and can never be
# recovered from the pools. The alternative -- truncating the 2051-wide row to 2048 --
# drops exactly that tail. select_k stays 512, so the indexer keeps its fused top-k fast
# path (sparse_attn_indexer_kpool.py:810 gates on select_k in {512,1024,2048}).
#
# Build (context = repo root):
#   docker build -f patches/Dockerfile.glm53-flash-sm120 -t vllm-glm53-flash-sm120:local .

ARG BASE=vllm/vllm-openai:glm53-flash@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4
FROM ${BASE}

RUN python3 - <<'PY'
from pathlib import Path

target = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "flashinfer_mla_sparse_sm120.py"
)
source = target.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch target: {old!r}")
    source = source.replace(old, new)


replace_once(
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n    is_sparse = True\n',
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n'
    "    is_sparse = True\n"
    "    supports_dense_mha_prefill = False\n",
)

replace_once(
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        from vllm.config import get_current_vllm_config\n",
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        self.rope_pad = 0\n"
    "        if self.qk_rope_head_dim == 0:\n"
    "            if self.kv_lora_rank != 512:\n"
    "                raise NotImplementedError(\n"
    '                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "\n'
    '                    "576-wide GLM_NSA geometry, which requires "\n'
    '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
    "                )\n"
    "            self.rope_pad = 64\n"
    "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
    "        from vllm.config import get_current_vllm_config\n",
)

replace_once(
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "        if self.rope_pad:\n"
    "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
)

replace_once(
    "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
    "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
)

replace_once(
    "        topk_indices_physical = cast(\n"
    "            torch.Tensor,\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "            ),\n"
    "        )\n",
    "        topk_indices_physical, topk_lengths = cast(\n"
    "            tuple[torch.Tensor, torch.Tensor],\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "                return_valid_counts=True,\n"
    "            ),\n"
    "        )\n"
    "        sparse_topk_capacity = topk_indices_physical.shape[1]\n"
    "        empty_rows = topk_lengths == 0\n"
    "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
    "            empty_rows, 0\n"
    "        )\n"
    "        topk_lengths = topk_lengths.clamp(min=1)\n",
)

replace_once(
    "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
    "            seq_lens=topk_lengths,\n            max_seq_len=sparse_topk_capacity,\n",
)

replace_once(
    "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
    "            sparse_mla_top_k=sparse_topk_capacity,\n",
)

replace_once(
    "        return out.squeeze(1), None\n",
    "        out = out.squeeze(1)\n"
    "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
    "        return out, None\n"
    "\n"
    "    def do_kv_cache_update(\n"
    "        self,\n"
    "        kv_c_normed: torch.Tensor,\n"
    "        k_pe: torch.Tensor,\n"
    "        kv_cache: torch.Tensor,\n"
    "        slot_mapping: torch.Tensor,\n"
    "        kv_cache_dtype: str,\n"
    "        k_scale: torch.Tensor,\n"
    "    ) -> None:\n"
    "        if self.rope_pad:\n"
    "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
    "        super().do_kv_cache_update(\n"
    "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
    "        )\n",
)

target.write_text(source)

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
old_width = "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)"
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    path = site / rel
    text = path.read_text()
    if text.count(old_width) != 1:
        raise RuntimeError(f"expected one buffer_width target in {rel}")
    path.write_text(text.replace(old_width, "buffer_width = topk_tokens"))

sparse_backend = site / "v1/attention/backends/mla/flashinfer_mla_sparse.py"
text = sparse_backend.read_text()
old_block_sizes = (
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
    "        return [64, 256]\n"
)
if text.count(old_block_sizes) != 1:
    raise RuntimeError("expected one SM120 kernel-block-size target")
sparse_backend.write_text(
    text.replace(
        old_block_sizes,
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64]\n",
    )
)

platform = site / "platforms/cuda.py"
text = platform.read_text()
old_align = "        return index_kpool * min(PAGED_MQA_PAGE_SIZES)\n"
if text.count(old_align) != 1:
    raise RuntimeError("expected one indexer block alignment target")
platform.write_text(
    text.replace(
        old_align,
        "        page_sizes = PAGED_MQA_PAGE_SIZES\n"
        "        capability = cls.get_device_capability()\n"
        "        if capability is not None and capability.major == 12:\n"
        "            page_sizes = tuple(p for p in page_sizes if p == 64)\n"
        "        return index_kpool * min(page_sizes)\n",
    )
)

indexer = site / "model_executor/layers/sparse_attn_indexer_kpool.py"
text = indexer.read_text()
for old, new in (
    (
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids, q_seq, index_kpool\n"
        "                    )\n",
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
        "                    )\n",
    ),
    (
        "            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
        "            out = expand_pools_and_append_tail(\n"
        "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
        "            )\n",
    ),
):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one expand-pools target: {old!r}")
    text = text.replace(old, new)
indexer.write_text(text)
PY

RUN python3 - <<'PY'
import inspect

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl as impl,
)

assert impl.supports_dense_mha_prefill is False
assert "do_kv_cache_update" in impl.__dict__
init_src = inspect.getsource(impl.__init__)
assert "self.rope_pad = 64" in init_src
fwd_src = inspect.getsource(impl.forward_mqa)
assert "torch.nn.functional.pad(q, (0, self.rope_pad))" in fwd_src
assert "qk_rope_head_dim=self.kernel_qk_rope_head_dim" in fwd_src
assert "return_valid_counts=True" in fwd_src
assert "sparse_mla_top_k=sparse_topk_capacity" in fwd_src
assert "seq_lens=topk_lengths" in fwd_src
assert "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)" in fwd_src
assert "attn_metadata.topk_tokens" not in fwd_src

from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    src = (site / rel).read_text()
    compile(src, rel, "exec")
    assert "buffer_width = topk_tokens\n" in src
    assert "kpool - 1 if kpool > 1" not in src
kpool_src = (site / "model_executor/layers/sparse_attn_indexer_kpool.py").read_text()
compile(kpool_src, "sparse_attn_indexer_kpool.py", "exec")
assert kpool_src.count("pool_ids[:, : select_k - 1]") == 2

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend as sm120_backend,
)

assert sm120_backend.get_supported_kernel_block_sizes() == [64]
align_src = (site / "platforms/cuda.py").read_text()
compile(align_src, "cuda.py", "exec")
assert "if capability is not None and capability.major == 12:" in align_src
assert "return index_kpool * min(page_sizes)" in align_src
print("glm53 NoPE sparse-MLA overlay verify OK")
PY
