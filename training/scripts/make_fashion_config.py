#!/usr/bin/env python3
"""Write configs/azyah_fashion_stage1.py: upstream Stage 1 with Azyah fashion data.

  python3 make_fashion_config.py <upstream training dir>

Data comes from data_tools.manifests output (AZ_MANIFEST_DIR/i2i.json = Glance/own
image pairs, AZ_MANIFEST_DIR/v2v.json = own clips [+ ViViD in research mode]).
Text-to-image/video tasks are switched off (no such data); probabilities and other
knobs come from env vars so one generated file serves every run:

  AZ_MANIFEST_DIR (required)   AZ_I2I_PROB (1.0)   AZ_V2V_PROB (1.0)
  AZ_EXP (fashion)  AZ_MAX_STEPS (2000)  AZ_CKPT_INTERVAL (250)  AZ_LR (2e-5)
  AZ_WARMUP (100)  AZ_GRAD_ACCUM (1)  AZ_MAX_FRAMES (49)  AZ_AUTO_RESUME (1)
  AZ_HSDP_SHARD (1)  AZ_SP_SIZE (1)  AZ_DETAILED_CAPTION_PROB (0.7)

Every replacement must match once, so an upstream change fails loudly.
"""
import sys
from pathlib import Path

E = 'field(default_factory=lambda: '
REPLACEMENTS = [
    ("TRAIN_RESOLUTION = (720, 1248)", "TRAIN_RESOLUTION = (480, 832)"),
    ("class Stage1ARConfig(ExpConfig):", "class AzyahFashionStage1Config(ExpConfig):"),
    ('    exp_name: str = "default"', f'    exp_name: str = {E}os.getenv("AZ_EXP", "fashion"))'),
    ("    auto_resume: bool = False", f'    auto_resume: bool = {E}os.getenv("AZ_AUTO_RESUME", "1") == "1")'),
    ('        default_factory=lambda: os.getenv("JOYAI_T2I_DATA", "data/t2i.json")',
     '        default_factory=lambda: None  # no text-to-image data in Azyah runs'),
    ('        default_factory=lambda: os.getenv("JOYAI_I2I_DATA", "data/i2i.json")',
     '        default_factory=lambda: _manifest("i2i")'),
    ('        default_factory=lambda: os.getenv("JOYAI_T2V_DATA", "data/t2v.json")',
     '        default_factory=lambda: None  # no text-to-video data in Azyah runs'),
    ('        default_factory=lambda: os.getenv("JOYAI_V2V_DATA", "data/v2v.json")',
     '        default_factory=lambda: _manifest("v2v")'),
    ("    image_sampling_prob: float = 1.0", "    image_sampling_prob: float = 0.0"),
    ("    multiple_images_sampling_prob: float = 2.0",
     f'    multiple_images_sampling_prob: float = {E}float(os.getenv("AZ_I2I_PROB", "1.0")) if _manifest("i2i") else 0.0)'),
    ("    video_sampling_prob: float = 1.0", "    video_sampling_prob: float = 0.0"),
    ("    multiple_videos_sampling_prob: float = 4.0",
     f'    multiple_videos_sampling_prob: float = {E}float(os.getenv("AZ_V2V_PROB", "1.0")) if _manifest("v2v") else 0.0)'),
    ("            min_temporal=49,\n            max_temporal=121,",
     '            min_temporal=49,\n            max_temporal=int(os.getenv("AZ_MAX_FRAMES", "49")),'),
    ("    max_train_steps: int = 10000000000", f'    max_train_steps: int = {E}int(os.getenv("AZ_MAX_STEPS", "2000")))'),
    ("    gradient_accumulation_steps: int = 1",
     f'    gradient_accumulation_steps: int = {E}int(os.getenv("AZ_GRAD_ACCUM", "1")))'),
    ("    checkpoint_interval: int = 500", f'    checkpoint_interval: int = {E}int(os.getenv("AZ_CKPT_INTERVAL", "250")))'),
    ("    learning_rate: float = 5e-5", f'    learning_rate: float = {E}float(os.getenv("AZ_LR", "2e-5")))'),
    ("    lr_warmup_steps: int = 0", f'    lr_warmup_steps: int = {E}int(os.getenv("AZ_WARMUP", "100")))'),
    ("    sp_size: int = 1", f'    sp_size: int = {E}int(os.getenv("AZ_SP_SIZE", "1")))'),
    ("    hsdp_shard_dim: int = 1", f'    hsdp_shard_dim: int = {E}int(os.getenv("AZ_HSDP_SHARD", "1")))'),
('    train_multiple_images_caption_keys: list[str] = field(\n        default_factory=lambda: ["caption"]\n    )',
     '    train_multiple_images_caption_keys: list[str] = field(\n        default_factory=lambda: ["caption", "caption_detailed"]\n    )'),
    ('    train_multiple_images_caption_sampling_prob: list[float] = field(\n        default_factory=lambda: [1.0]\n    )',
     '    train_multiple_images_caption_sampling_prob: list[float] = field(\n        default_factory=lambda: _caption_probs()\n    )'),
    ('    train_multiple_videos_caption_keys: list[str] = field(\n        default_factory=lambda: ["caption"]\n    )',
     '    train_multiple_videos_caption_keys: list[str] = field(\n        default_factory=lambda: ["caption", "caption_detailed"]\n    )'),
    ('    train_multiple_videos_caption_sampling_prob: list[float] = field(\n        default_factory=lambda: [1.0]\n    )',
     '    train_multiple_videos_caption_sampling_prob: list[float] = field(\n        default_factory=lambda: _caption_probs()\n    )'),
]

HELPER = '''

def _manifest(group):
    """Path to an Azyah manifest (from data_tools.manifests), or None if that group has no data."""
    root = os.getenv("AZ_MANIFEST_DIR")
    if not root:
        raise RuntimeError("set AZ_MANIFEST_DIR to the folder written by data_tools.manifests")
    p = os.path.join(root, f"{group}.json")
    return p if os.path.exists(p) else None


def _caption_probs():
    """Short live caption vs detailed live caption (review R2). Samples without a
    detailed caption always use the short one (the loader skips missing keys)."""
    d = float(os.getenv("AZ_DETAILED_CAPTION_PROB", "0.7"))
    return [1.0 - d, d]
'''


def main(training_dir: str) -> Path:
    src = Path(training_dir) / "configs" / "stage1_ar.py"
    text = src.read_text()
    for old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"upstream config changed: expected 1 match, found {n}: {old[:70]!r}")
        text = text.replace(old, new)
    anchor = "\n@dataclass\nclass AzyahFashionStage1Config"
    if text.count(anchor) != 1:
        sys.exit("cannot place the manifest helper: upstream layout changed")
    text = text.replace(anchor, HELPER + anchor)
    header = '"""Azyah fashion Stage 1 (generated by fluxrt-serverless/training/scripts/make_fashion_config.py)."""\n'
    out = src.parent / "azyah_fashion_stage1.py"
    out.write_text(header + text)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
