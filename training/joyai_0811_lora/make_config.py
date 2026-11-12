#!/usr/bin/env python3
"""Write configs/azyah_joyai0811_lora.py: a LoRA on the live joyai_video_edit_dit_0811.pth,
trained through the live 2-step streaming rollout (train.py), with the live input format.

  python3 make_config.py <upstream training dir>

Starts from the same upstream stage1_ar.py and the same Azyah replacements as
scripts/make_fashion_config.py (manifests, LoRA fields, caption keys, resume), then switches
everything that differs between training defaults and the live session:

  resolution 480x840 (the live JOYAI_SESSION_WIDTH x HEIGHT; the VAE's 24-pixel factor divides both)
  static clips of AZ_STATIC_FRAMES (25 = 4 latent chunks: sink + the 3-chunk window)
  garment reference basesize 768 with upstream's default buckets (as deploy/xvideo/config.py)
  2 steps, no CFG (guidance 1.0, source guidance 1.0, cfg_rate 0: the live session never blanks the text)
  the rollout knobs train.py hands to upstream generate_dmd_latents: 2 steps, chunk 1,
  window/sink from the model config, max_temporal_ids 8 (the live absolute-id path)

The base checkpoint is JOYAI_BASE_MODEL (run_train.sh points it at the 0811 clone on the
training volume). Env knobs: everything make_fashion_config reads, plus AZ_STATIC_FRAMES (25),
AZ_EXIT (random|last). Every replacement must match exactly once, so an upstream change fails loudly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

E = 'field(default_factory=lambda: '

EXTRA = [
    ("TRAIN_RESOLUTION = (480, 832)",
     "TRAIN_RESOLUTION = (480, 840)  # live 480x840 portrait (both orientations get a bucket)"),
    ("class AzyahFashionStage1Config(ExpConfig):", "class AzyahJoyAI0811LoraConfig(ExpConfig):"),
    ('            min_temporal=49,\n            max_temporal=int(os.getenv("AZ_MAX_FRAMES", "49")),',
     '            min_temporal=int(os.getenv("AZ_STATIC_FRAMES", "25")),\n'
     '            max_temporal=int(os.getenv("AZ_STATIC_FRAMES", "25")),'),
    ("    ref_image_basesize: int = TRAIN_RESOLUTION[0]",
     "    ref_image_basesize: int = 768  # = live (deploy/xvideo/config.py ref_image_basesize)"),
    ("    ref_image_bucket_configs: list[tuple[int, int, int, int, int]] = field(\n"
     "        default_factory=lambda: generate_video_image_bucket(\n"
     "            img_basesizes=[TRAIN_RESOLUTION],\n"
     "            spatial_multiple=24,\n"
     "            bs_img=1,\n"
     "            bs_vid=0,\n"
     "            bs_mimg=0,\n"
     "            bs_mvid=0,\n"
     "        )\n"
     "    )",
     "    # None: the loader builds generate_video_image_bucket(img_basesize=768, bs_img=1), the buckets\n"
     "    # the live session resizes the garment photo to.\n"
     "    ref_image_bucket_configs: list[tuple[int, int, int, int, int]] | None = None"),
    ("    cfg_rate: float = 0.01", "    cfg_rate: float = 0.0  # live never blanks the instruction"),
    ("    ref_cfg_rate: float = 0.01", "    ref_cfg_rate: float = 0.0"),
    ("    num_inference_steps: int = 30", "    num_inference_steps: int = 2  # = JOYAI_STEPS"),
    ("    guidance_scale: float = 3.5", "    guidance_scale: float = 1.0  # no CFG, as live"),
    ("    source_guidance_scale: float = 1.5", "    source_guidance_scale: float = 1.0"),
]

ANCHOR = '    stage1_lora: bool = field(default_factory=lambda: os.getenv("AZ_STAGE1_LORA", "1") == "1")'
ROLLOUT_FIELDS = '''
    # Azyah 0811 LoRA (joyai_0811_lora/train.py): upstream generate_dmd_latents runs the live
    # 2-step chunked rollout with these; the LoRA is the stage1_lora "generator" adapter.
    dmd_use_lora: bool = True
    dmd_generator_lora_adapter_name: str = "generator"
    dmd_generator_num_inference_steps: int = 2
    dmd_generator_chunk_size: int | None = None  # the model's chunk_size (1), as live
    dmd_generator_window_size: int | None = None  # the model's local_window_size (3), as live
    dmd_generator_global_sink_chunk: bool | None = None  # the model's global_sink_chunk, as live
    dmd_rollout_num_chunks: int = -1  # the whole clip
    dmd_rollout_use_relative_temporal_ids: bool = True  # with max_temporal_ids: the live id path
    max_temporal_ids: int = 8  # = JOYAI_MAX_TEMPORAL_IDS
    sft_exit: str = field(default_factory=lambda: os.getenv("AZ_EXIT", "random"))'''


def _fashion_module():
    here = Path(__file__).resolve().parent
    for p in (os.getenv("AZ_FASHION_CONFIG_PY"), here.parent / "scripts" / "make_fashion_config.py"):
        if p and Path(p).exists():
            spec = importlib.util.spec_from_file_location("make_fashion_config", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    sys.exit("make_fashion_config.py not found (set AZ_FASHION_CONFIG_PY)")


def build(text: str, fashion) -> str:
    for old, new in [*fashion.REPLACEMENTS, *EXTRA]:
        n = text.count(old)
        if n != 1:
            sys.exit(f"upstream config changed: expected 1 match, found {n}: {old[:70]!r}")
        text = text.replace(old, new)
    if text.count(ANCHOR) != 1:
        sys.exit("cannot place the rollout fields: the stage1_lora field is missing")
    text = text.replace(ANCHOR, ANCHOR + ROLLOUT_FIELDS)
    anchor = "\n@dataclass\nclass AzyahJoyAI0811LoraConfig"
    if text.count(anchor) != 1:
        sys.exit("cannot place the manifest helper: upstream layout changed")
    return text.replace(anchor, fashion.HELPER + anchor)


def main(training_dir: str) -> Path:
    src = Path(training_dir) / "configs" / "stage1_ar.py"
    text = build(src.read_text(), _fashion_module())
    header = '"""Azyah 0811 LoRA (generated by fluxrt-serverless/training/joyai_0811_lora/make_config.py)."""\n'
    out = src.parent / "azyah_joyai0811_lora.py"
    out.write_text(header + text)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
