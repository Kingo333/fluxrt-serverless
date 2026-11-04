"""P9e: generate try-on clips for held-out ViViD test pairs with one model, offline.

Run once per model (the Stage 1 starting point and the merged exp1), from the upstream
training folder, then score both with eval.run_eval:

  cd $AZ_UPSTREAM/training && PYTHONPATH=$PWD:$AZYAH_TRAINING torchrun --nproc_per_node 1 \\
    -m eval.generate_vivid --config configs/azyah_fashion_stage1.py --ckpt MODEL.pth --name exp1 \\
    --staging /workspace/azyah-vton/data/staging/vivid --out /workspace/azyah-vton/eval/p9e --max-items 24

For each test pair (split=test, pseudo-source present, sorted by key, first N):
  source = the JoyAI-made pseudo-source clip (what a camera would see: person in a different
           garment), garment = the real product photo, prompt = the training caption.
Writes OUT/<key>/input.mp4, ref.jpg, target.mp4 (real ground truth, once) and out_<name>.mp4,
the layout eval/run_eval.py expects. Same seed, frames and size for every model, so the
only difference between out_base and out_exp1 is the weights.

Uses upstream evaluation/benchmark/common.py (load_reference_video, build_pipeline_kwargs) and
adds the garment photo as ref_image_latent, encoded the way training does (train_t2v.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def test_items(staging: Path, max_items: int) -> list[Path]:
    out = []
    for d in sorted(p for p in staging.iterdir() if p.is_dir()):
        meta_path = d / "meta.json"
        if not meta_path.exists() or not (d / "source.mp4").exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("split") == "test" and (meta.get("pseudo_source") or {}).get("ok", True):
            out.append(d)
        if len(out) >= max_items:
            break
    return out


# Same defaults as training/scripts/common.sh. The config reads these; unset, it falls back to
# relative "checkpoints/..." paths and transformers tries them as Hub repos (P9e: HF 401 for
# checkpoints/vae).
MODEL_ENV = {
    "JOYAI_BASE_MODEL": "dit/pretrain_editing_bid_version.pth",
    "JOYAI_VAE": "vae",
    "JOYAI_TEXT_ENCODER": "MiMo-VL-7B-RL-2508",
}


def default_model_env(env=os.environ) -> dict:
    """Fill unset JOYAI_* paths from $AZ_ROOT/checkpoints; fail early if one is missing."""
    ckpt = Path(env.get("AZ_ROOT", "/workspace/azyah-vton")) / "checkpoints"
    for k, sub in MODEL_ENV.items():
        env.setdefault(k, str(ckpt / sub))
    missing = [f"{k}={env[k]}" for k in MODEL_ENV if not Path(env[k]).exists()]
    if missing:
        raise SystemExit("model files not found: " + ", ".join(missing))
    return {k: env[k] for k in MODEL_ENV}


def _ref_latent(pipeline, cfg, ref_path: Path, height: int, width: int, device, dtype):
    import torch
    from PIL import Image
    im = Image.open(ref_path).convert("RGB").resize((width, height), Image.BICUBIC)
    x = torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1).float().div(127.5).sub(1.0)
    x = x.unsqueeze(1).unsqueeze(0).to(device=device, dtype=dtype)  # [1, c, t=1, h, w]
    with torch.inference_mode():
        if cfg.enable_denormalization:
            lat = pipeline.normalize_latents(pipeline.vae.encode(x).latent_dist.sample())
        else:
            lat = pipeline.vae.encode(x)
    return lat[:, :, :1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="model label, e.g. base or exp1")
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=480)
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    print("model paths:", default_model_env())
    Path(a.out).mkdir(parents=True, exist_ok=True)

    import shutil
    import torch
    from evaluation.benchmark.common import BenchmarkItem, build_pipeline_kwargs
    from src.config import load_config_class_from_pyfile
    from src.distributed.parallel_states import clean_dist_env, init_distributed_environment_and_sequence_parallel
    from src.models import load_dit, load_pipeline
    from src.utils import extract_video_tensor, save_video, seed_everything
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_tools.staging import instruction

    cfg = load_config_class_from_pyfile(a.config)()
    cfg.dit_ckpt = a.ckpt
    cfg.hsdp_shard_dim = 1
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    init_distributed_environment_and_sequence_parallel(1)
    try:
        seed_everything(a.seed)
        model = load_dit(cfg, device=device)
        model.config.use_inference_kv_cache = True
        model.requires_grad_(False).eval()
        pipeline = load_pipeline(cfg, model, device)
        pipeline.set_progress_bar_config(disable=True)
        args = SimpleNamespace(resolution=a.resolution, num_frames=a.num_frames, neg_prompt="",
                               num_inference_steps=None, guidance_scale=None, source_guidance_scale=1.0,
                               use_relative_rope=True, max_temporal_ids=8, store_clean_only_self=True)
        out_root = Path(a.out)
        done = 0
        for d in test_items(Path(a.staging), a.max_items):
            meta = json.loads((d / "meta.json").read_text())
            dest = out_root / d.name
            result_path = dest / f"out_{a.name}.mp4"
            if result_path.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            item = BenchmarkItem(meta.get("category", "upper"), instruction(meta.get("category", "upper")), d / "source.mp4")
            generator = torch.Generator(device=device).manual_seed(a.seed)
            kwargs, source_pixels = build_pipeline_kwargs(item, args, cfg, generator)
            ref = next(p for p in d.iterdir() if p.stem == "ref")
            kwargs["ref_image_latent"] = _ref_latent(pipeline, cfg, ref, kwargs["height"], kwargs["width"],
                                                     device, next(model.parameters()).dtype)
            with torch.inference_mode():
                result = pipeline(**kwargs)
            save_video(extract_video_tensor(result[0, -1]), str(result_path), fps=cfg.fps)
            if not (dest / "input.mp4").exists():
                save_video(extract_video_tensor(source_pixels), str(dest / "input.mp4"), fps=cfg.fps)
                shutil.copy(ref, dest / f"ref{ref.suffix}")
                shutil.copy(d / "target.mp4", dest / "target.mp4")
            done += 1
            print(json.dumps({"key": d.name, "model": a.name, "done": done}), flush=True)
    finally:
        clean_dist_env()


if __name__ == "__main__":
    main()
