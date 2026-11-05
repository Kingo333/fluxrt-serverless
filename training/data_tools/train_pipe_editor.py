"""The live 0811 edit through JoyAI's TRAINING streaming pipeline (src.models.load_pipeline),
so step 3b can run on the plain RunPod PyTorch template with the training venv on the volume,
without the large endpoint image (its pull hung three Pods in a row, 2026-11-05).

Same settings as the live session and as step 1 (generate_vivid --live, which the owner rated
"similar to the app"): chunk-1 causal streaming, 2 steps, guidance 1, relative temporal ids with
max_temporal_ids 8, the garment photo as a fixed reference latent (bucket basesize 768), and the
app's instruction read with the first camera frame. Differences vs the live server: bf16 instead
of FP8, and the whole clip is processed in one call instead of frame by frame over a socket.

Run under torchrun (one process), from the upstream training folder on PYTHONPATH:
  torchrun --nproc_per_node 1 -m data_tools.live_source ... --engine trainpipe --config CONFIG.py
"""

from __future__ import annotations

TEMPORAL = 8  # VAE_TEMPORAL_FACTOR: clip lengths are 8n+1


def clip_len(n: int) -> int:
    """Frames the pipeline takes from n input frames: 8k+1, at least 9 (stills are padded)."""
    return max(9, ((n - 1) // TEMPORAL) * TEMPORAL + 1)


def fit_frames(frames: list, n_out: int) -> list:
    """Pad (repeat the last frame) or trim to n_out frames."""
    out = list(frames[:n_out])
    while len(out) < n_out:
        out.append(out[-1])
    return out


class TrainPipeEditor:
    name = "joyai-0811-trainpipe"

    def __init__(self, dit: str, config: str, steps: int = 2, seed: int = 42, ref_basesize: int = 768):
        import torch
        from src.config import load_config_class_from_pyfile
        from src.distributed.parallel_states import init_distributed_environment_and_sequence_parallel
        from src.models import load_dit, load_pipeline
        cfg = load_config_class_from_pyfile(config)()
        cfg.dit_ckpt = dit
        cfg.hsdp_shard_dim = 1
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        init_distributed_environment_and_sequence_parallel(1)
        model = load_dit(cfg, device=self.device)
        model.config.use_inference_kv_cache = True
        model.requires_grad_(False).eval()
        self.dtype = next(model.parameters()).dtype
        self.pipe = load_pipeline(cfg, model, self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg, self.steps, self.seed, self.ref_basesize = cfg, steps, seed, ref_basesize

    def _ref_latent(self, ref_image, h: int, w: int):
        import numpy as np
        import torch
        from src.config import generate_video_image_bucket
        from src.utils import _dynamic_resize_from_bucket
        cfgs = generate_video_image_bucket(img_basesize=self.ref_basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0)
        im, _ = _dynamic_resize_from_bucket(ref_image.convert("RGB"), bucket_configs=cfgs, num_frames=1, num_items=1,
                                            return_bucket=True)
        if torch.is_tensor(im):
            px = im.clamp(0, 255).to(torch.uint8).reshape(-1, *im.shape[-2:])[:3]
        else:
            px = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        x = px.float().div(127.5).sub(1.0).unsqueeze(1).unsqueeze(0).to(self.device, self.dtype)
        with torch.inference_mode():
            if self.cfg.enable_denormalization:
                lat = self.pipe.normalize_latents(self.pipe.vae.encode(x).latent_dist.sample())
            else:
                lat = self.pipe.vae.encode(x)
        return lat[:, :, :1]

    def edit(self, frames, prompt, size, ref_image=None):
        import numpy as np
        import torch
        from PIL import Image
        from src.utils import extract_video_tensor
        w, h = size
        n = len(frames)
        t = clip_len(n)
        use = [f.convert("RGB").resize((w, h)) for f in fit_frames(frames, t)]
        px = torch.from_numpy(np.stack([np.asarray(f) for f in use]))            # t h w c, uint8
        ref_video = px.float().div(127.5).sub(1.0).permute(3, 0, 1, 2).unsqueeze(0)  # 1 c t h w
        vit = self.cfg.use_vit
        kwargs = dict(
            prompt=[f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n" if vit else prompt],
            negative_prompt=["<|im_start|>user\n<image>\n<|im_end|>\n" if vit else ""],
            images=[use[0]] if vit else None,
            reference_visual_content=ref_video.unsqueeze(1),
            height=h, width=w, num_frames=t,
            num_inference_steps=self.steps, guidance_scale=1.0, source_guidance_scale=1.0,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
            output_type="pt", return_dict=False,
            enable_denormalization=self.cfg.enable_denormalization,
            max_sequence_length=self.cfg.text_token_max_length, use_vit=vit,
            use_relative_temporal_ids=True, store_clean_only_self=True, max_temporal_ids=8,
        )
        if ref_image is not None:
            kwargs["ref_image_latent"] = self._ref_latent(ref_image, h, w)
        with torch.inference_mode():
            result = self.pipe(**kwargs)
        video = extract_video_tensor(result[0, -1])                            # t c h w, uint8
        out = [Image.fromarray(video[k].permute(1, 2, 0).numpy()) for k in range(video.shape[0])]
        return fit_frames(out, n)
