#!/usr/bin/env python3
"""CPU pre-check of the real upstream training data loader (no GPU, no model).

  cd $AZ_UPSTREAM/training && PYTHONPATH=$PWD $AZ_ROOT/venv/bin/python \\
      /path/check_dataloader.py configs/azyah_smoke_stage1.py [--batches 6]

Builds the loader exactly as src/train_t2v.py does (build_webdataset_dataloader with the
config's data fields) and pulls a few batches, so missing decoders, bad shards or
caption/feature problems show up on a cheap CPU Pod instead of a GPU smoke run.
Prints one line per batch and DATALOADER_OK at the end.
"""

import argparse
import sys
import time


def describe(batch) -> str:
    import torch
    if isinstance(batch, dict):
        parts = []
        for k, v in batch.items():
            if torch.is_tensor(v):
                parts.append(f"{k}={tuple(v.shape)}")
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                parts.append(f"{k}=[{len(v)}x{tuple(v[0].shape)}]")
            elif isinstance(v, (list, tuple)):
                parts.append(f"{k}=list[{len(v)}]")
            else:
                parts.append(f"{k}={type(v).__name__}")
        return " ".join(parts)
    return type(batch).__name__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args(argv)
    from src.config import load_config_class_from_pyfile
    from src.dataset.webdataset import build_webdataset_dataloader
    cfg = load_config_class_from_pyfile(a.config)()
    loader = build_webdataset_dataloader(
        image_data_files=cfg.train_image_data_files,
        multiple_images_data_files=cfg.train_multiple_images_data_files,
        multiple_videos_data_files=cfg.train_multiple_videos_data_files,
        video_data_files=cfg.train_video_data_files,
        bucket_configs=cfg.bucket_configs,
        bucket_configs_options=cfg.bucket_configs_options,
        bucket_configs_options_prob=cfg.bucket_configs_options_prob,
        prioritize_frame_matching=cfg.prioritize_frame_matching,
        image_caption_keys=cfg.train_image_caption_keys,
        image_caption_sampling_prob=cfg.train_image_caption_sampling_prob,
        multiple_images_caption_keys=cfg.train_multiple_images_caption_keys,
        multiple_images_caption_sampling_prob=cfg.train_multiple_images_caption_sampling_prob,
        multiple_videos_caption_keys=cfg.train_multiple_videos_caption_keys,
        multiple_videos_caption_sampling_prob=cfg.train_multiple_videos_caption_sampling_prob,
        video_caption_keys=cfg.train_video_caption_keys,
        video_caption_sampling_prob=cfg.train_video_caption_sampling_prob,
        image_sampling_prob=cfg.image_sampling_prob,
        multiple_images_sampling_prob=cfg.multiple_images_sampling_prob,
        multiple_videos_sampling_prob=cfg.multiple_videos_sampling_prob,
        video_sampling_prob=cfg.video_sampling_prob,
        ensure_divisible_shards=cfg.ensure_divisible_shards,
        shuffle=cfg.shuffle,
        num_workers=0,
        fps=cfg.fps,
        seed=cfg.seed,
        tar_files_shuffle_seed=cfg.tar_files_shuffle_seed,
        rec_aug_rate=cfg.rec_aug_rate,
        ref_image_basesize=cfg.ref_image_basesize,
        ref_image_bucket_configs=cfg.ref_image_bucket_configs,
    )
    t0 = time.time()
    for i, batch in enumerate(loader):
        print(f"batch {i}: {describe(batch)} ({time.time() - t0:.1f}s)", flush=True)
        if i + 1 >= a.batches:
            break
    print("DATALOADER_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
