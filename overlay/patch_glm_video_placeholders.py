# Align GLM-5.3 video prompt placeholders to encoder grid_t.
# glm4_1v builds one image-block per timestamp; a 1s clip can yield 3 timestamps
# while video_grid_thw T=1, so vLLM assigns 100 encoder tokens to 300 slots.
from __future__ import annotations


def _align_timestamps(timestamps: list, t_groups: int) -> list:
    if t_groups < 1:
        t_groups = 1
    if not timestamps:
        return [0] * t_groups
    n = len(timestamps)
    if n == t_groups:
        return timestamps
    if t_groups == 1:
        return [timestamps[0]]
    if n < t_groups:
        return timestamps + [timestamps[-1]] * (t_groups - n)
    last = n - 1
    return [timestamps[int(round(i * last / (t_groups - 1)))] for i in range(t_groups)]


def apply() -> None:
    from vllm.model_executor.models.glm4_1v import (
        Glm4vProcessingInfo,
        Glm4vProcessor,
    )

    if getattr(Glm4vProcessingInfo, "_glm53_video_t_aligned", False):
        return

    def _construct_video_placeholder(self, video_array, metadata, grid_thw):
        hf_processor = self.get_hf_processor()
        tokenizer = self.get_tokenizer()
        image_processor = hf_processor.image_processor
        hf_config = self.get_hf_config()
        merge_length = image_processor.merge_size**2
        t_hw = grid_thw.reshape(-1)
        t_groups, height, width = int(t_hw[0]), int(t_hw[1]), int(t_hw[2])
        n_per = int(height * width) // merge_length

        # Glm5NextVideoProcessor has no max_duration; glm4v timestamps crash.
        name = type(hf_processor).__name__
        if isinstance(hf_processor, Glm4vProcessor) and "Glm5" not in name:
            timestamps = self._get_video_second_idx_glm4v(metadata, len(video_array))
            ts_fmt = "{}"
        elif self._is_glmga_model(hf_processor):
            timestamps = self._get_video_second_idx_glmga(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"
        else:
            timestamps = self._get_video_second_idx_glm46v(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"

        timestamps = _align_timestamps(list(timestamps), t_groups)
        embed_id = self._get_video_frame_embed_token_id(hf_processor)
        placeholder = [hf_config.video_start_token_id]
        for ts in timestamps:
            placeholder.append(hf_config.image_start_token_id)
            placeholder.extend([embed_id] * n_per)
            placeholder.append(hf_config.image_end_token_id)
            placeholder.extend(
                tokenizer.encode(ts_fmt.format(ts), add_special_tokens=False)
            )
        placeholder.append(hf_config.video_end_token_id)
        return placeholder

    Glm4vProcessingInfo._construct_video_placeholder = _construct_video_placeholder
    Glm4vProcessingInfo._glm53_video_t_aligned = True


def _install_import_hook() -> None:
    import builtins
    import importlib

    if getattr(builtins, "_glm53_video_hook", False):
        return
    builtins._glm53_video_hook = True
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = real_import(name, globals, locals, fromlist, level)
        if "glm4_1v" in name:
            try:
                apply()
            except Exception:
                pass
        return mod

    builtins.__import__ = _import
    real_im = importlib.import_module

    def _im(name, package=None):
        mod = real_im(name, package)
        if "glm4_1v" in name:
            try:
                apply()
            except Exception:
                pass
        return mod

    importlib.import_module = _im


_install_import_hook()
try:
    apply()
except Exception:
    pass


def _disable_gb10_persistent_topk() -> None:
    """Decode-path persistent_topk oversubscribes GB10 smem on long seqs."""
    from pathlib import Path

    p = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "sparse_attn_indexer_kpool.py"
    )
    if not p.is_file():
        return
    text = p.read_text()
    old = "if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
    new = (
        "if False and current_platform.is_cuda() and "
        "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem"
    )
    if old in text:
        p.write_text(text.replace(old, new, 1))


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve()
    dst = Path("/usr/local/lib/python3.12/dist-packages/glm53_video_patch.py")
    shutil.copy(src, dst)
    Path("/usr/local/lib/python3.12/dist-packages/glm53_video.pth").write_text(
        "import glm53_video_patch\n"
    )
    _disable_gb10_persistent_topk()
