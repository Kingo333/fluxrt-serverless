"""0811 LoRA trainer: a LoRA on the live joyai_video_edit_dit_0811.pth, trained through the
same 2-step chunked rollout the live session runs, against the REAL target.

  cd $AZ_UPSTREAM/training && torchrun --nproc_per_node 1 -m src.azyah_train_0811_lora \
      --config configs/azyah_joyai0811_lora.py
(run_train.sh copies this file to training/src/azyah_train_0811_lora.py.)

Per step (batch 1):
  1. prepare_inputs (upstream train_t2v): VAE latents of the person clip (source) and the real
     target clip, the garment photo as ref_image_latent, the instruction + first camera frame
     through the text encoder (use_vit), exactly as Stage 1/Stage 2 training feed the model.
  2. generate_dmd_latents (upstream train_t2v_dmd, the rollout JoyAI used to distil 0811):
     for every latent chunk, in order, with the first chunk as sink and the recent window as
     clean history (time 0), the live scheduler's 2 steps (t = 1000, 837.64). At one of the two
     steps (AZ_EXIT: random = upstream, last = the served output) it returns the x0 prediction
     with gradient; the history chunks are its own earlier outputs, detached (self-forcing).
  3. loss = garment-weighted MSE(x0 prediction, real target latents) + AZ_EDGE_WEIGHT x edge L1
     inside the garment (losses.py; mask = the staged Glance mask, cropped like the clip and
     written by make_static_clips to AZ_MASK_DIR/<sample key>.png; no mask = weight 1 everywhere).
     Only the LoRA ("generator" adapter, base frozen by patches/stage1-lora.patch) gets gradients.

So the LoRA is only ever asked to make the live 2-step output closer to the real photo; it is
never trained at other noise levels or step counts, which is what would undo the distillation.
Saves use the patched Stage 1 LoRA format (generator_lora + config; refuses empty saves and
resumes) and are never pruned: every AZ_CKPT_INTERVAL save is evaluated against 0811.
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import src.train_t2v_dmd as dmd
from src.config import ExpConfig, load_config_class_from_pyfile
from src.dataset.webdataset import build_webdataset_dataloader
from src.distributed.communications import broadcast_item
from src.distributed.parallel_states import clean_dist_env
from src.models import load_pipeline
from src.train_t2v import prepare_inputs, setup_distributed_training, setup_model_and_optimizer
from src.utils import find_latest_checkpoint, seed_everything
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import setup_logger

sys.path.insert(0, os.getenv("AZ_TOOLS_DIR", str(Path(__file__).resolve().parent)))
from live_schedule import (  # noqa: E402
    LIVE_MAX_TEMPORAL_IDS, LIVE_REF_BASESIZE, LIVE_STEPS, choose_exit, live_timesteps, matches_live,
)
from losses import garment_loss, mask_to_grid  # noqa: E402

MASK_WEIGHT = float(os.getenv("AZ_MASK_WEIGHT", "3.0"))
EDGE_WEIGHT = float(os.getenv("AZ_EDGE_WEIGHT", "0.5"))


def load_mask(key: str, h: int, w: int, device):
    """The garment mask of one sample on the latent grid, or None (no mask = weight 1)."""
    d = os.getenv("AZ_MASK_DIR")
    if not d:
        return None
    p = Path(d) / f"{key}.png"
    if not p.exists():
        return None
    from PIL import Image
    return torch.from_numpy(mask_to_grid(Image.open(p), h, w)).to(device=device, dtype=torch.float32)


def self_check(cfg: ExpConfig, pipeline, model, logger) -> None:
    """Fail before step 1 if anything differs from the live session."""
    problems = []
    pipeline.scheduler.set_timesteps(int(cfg.dmd_generator_num_inference_steps), device="cpu")
    if not matches_live(pipeline.scheduler.timesteps.tolist(), LIVE_STEPS):
        problems.append(f"timesteps {pipeline.scheduler.timesteps.tolist()} != live {live_timesteps()}")
    mc = model.config
    if getattr(mc, "chunk_size", None) != 1 or not getattr(mc, "causal", False):
        problems.append(f"model chunk_size={getattr(mc, 'chunk_size', None)} causal={getattr(mc, 'causal', None)}; live is 1/causal")
    if int(getattr(cfg, "max_temporal_ids", 0) or 0) != LIVE_MAX_TEMPORAL_IDS:
        problems.append(f"max_temporal_ids={cfg.max_temporal_ids}; live is {LIVE_MAX_TEMPORAL_IDS}")
    if int(cfg.ref_image_basesize) != LIVE_REF_BASESIZE or cfg.ref_image_bucket_configs is not None:
        problems.append("garment reference is not resized like live (basesize 768, default buckets)")
    if float(cfg.cfg_rate) != 0.0 or not cfg.use_vit:
        problems.append(f"cfg_rate={cfg.cfg_rate} use_vit={cfg.use_vit}; live never blanks the text and uses the frame")
    if not getattr(cfg, "stage1_lora", False) or not getattr(model, "_azyah_stage1_lora", False):
        problems.append("the stage1-lora patch is not active (no frozen base + generator adapter)")
    if problems:
        raise SystemExit("0811 LoRA self-check failed:\n  " + "\n  ".join(problems))
    logger.info(f"0811 LoRA self-check ok: steps {live_timesteps()}, chunk 1 causal, window "
                f"{getattr(mc, 'local_window_size', '?')} sink {getattr(mc, 'global_sink_chunk', '?')}, "
                f"max_temporal_ids {cfg.max_temporal_ids}, ref basesize {cfg.ref_image_basesize}, exit {cfg.sft_exit}")


def install_exit_policy(cfg: ExpConfig, rng: random.Random) -> None:
    """generate_dmd_latents asks dmd._sample_exit_index which step to back-propagate through."""
    mode = str(getattr(cfg, "sft_exit", "random"))
    choose_exit(mode, LIVE_STEPS, rng)  # validates the mode
    dmd._sample_exit_index = lambda n: choose_exit(mode, int(n), rng)


def build_dataloader(cfg: ExpConfig):
    return build_webdataset_dataloader(
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
        num_workers=cfg.num_workers,
        fps=cfg.fps,
        seed=cfg.seed,
        tar_files_shuffle_seed=cfg.tar_files_shuffle_seed,
        rec_aug_rate=cfg.rec_aug_rate,
        ref_image_basesize=cfg.ref_image_basesize,
        ref_image_bucket_configs=cfg.ref_image_bucket_configs,
    )


def main(cfg: ExpConfig) -> None:
    _local_rank, global_rank, world_size, device = setup_distributed_training(cfg)
    if world_size != 1:
        raise SystemExit("the 0811 LoRA trainer is single-GPU (GPUS=1)")
    seed_everything(cfg.seed)
    exp_dir = Path(cfg.output_dir) / f"{cfg.exp_name}_sp{cfg.sp_size}_world{world_size}"
    log_dir, ckpt_dir = exp_dir / "logs", exp_dir / "checkpoints"
    for d in (exp_dir, log_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = setup_logger(log_dir)
    tb = SummaryWriter(log_dir=log_dir)
    (exp_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(cfg.to_json_string())

    loader = build_dataloader(cfg)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(cfg, device)  # patched: frozen base + LoRA

    init_step, epoch = 0, 0
    latest = find_latest_checkpoint(ckpt_dir) if cfg.auto_resume else None
    if latest is not None:
        logger.info(f"Auto-resume from {latest}")
        init_step, epoch = load_checkpoint(model=model, path=latest, device=device, optimizer=optimizer,
                                           scheduler=lr_scheduler, dataloader=loader)
    pipeline = load_pipeline(cfg, model, device)
    self_check(cfg, pipeline, model, logger)

    noise_seed = cfg.seed + global_rank + init_step
    noise_cuda = torch.Generator(device="cuda").manual_seed(noise_seed)
    rng = random.Random(noise_seed)
    install_exit_policy(cfg, rng)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_masks = len(list(Path(os.environ["AZ_MASK_DIR"]).glob("*.png"))) if os.getenv("AZ_MASK_DIR") else 0
    logger.info(f"0811 LoRA loss: garment weight {MASK_WEIGHT} inside the mask, edge weight {EDGE_WEIGHT}, "
                f"{n_masks} masks in {os.getenv('AZ_MASK_DIR') or '(none)'}")
    logger.info(f"0811 LoRA: {sum(p.numel() for p in trainable) / 1e6:.1f}M trainable, steps {init_step + 1}.."
                f"{cfg.max_train_steps}, LR {cfg.learning_rate}, saves every {cfg.checkpoint_interval} (all kept)")

    step_times: deque[float] = deque(maxlen=50)
    it = iter(loader)
    gc.disable()
    for step in range(init_step + 1, cfg.max_train_steps + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, used, mse_sum, edge_sum, masked = 0.0, 0, 0.0, 0.0, 0
        for _ in range(cfg.gradient_accumulation_steps):
            batch = next(it)
            latents, prompt_embeds, prompt_mask, extra = prepare_inputs(
                pixel_values=batch["pixel"], caption=batch["caption"], pipeline=pipeline, cfg=cfg, rng=rng,
                generator=noise_cuda, ref_image_pixels=batch.get("ref_image_pixel"))
            if extra["task_type"] != "v2v":
                logger.warning(f"step {step}: skipping a {extra['task_type']} sample (live is video)")
                continue
            target, _ = dmd._flatten_multi_item_latents(latents, extra)
            pred, _, _ = dmd.generate_dmd_latents(
                cfg, pipeline, model, target, prompt_embeds, prompt_mask, extra, noise_cuda,
                requires_grad=True, rollout_num_chunks_override=-1, group_backward_fn=None, controller=None)
            keys = batch.get("key") or [""]
            mask = load_mask(str(keys[0]), int(target.shape[-2]), int(target.shape[-1]), target.device)
            terms = garment_loss(pred.float(), target.float(), mask, MASK_WEIGHT, EDGE_WEIGHT)
            masked += int(mask is not None)
            mse_sum += float(terms["mse"].detach())
            edge_sum += float(terms["edge"].detach())
            loss = terms["total"] / cfg.gradient_accumulation_steps
            if not torch.isfinite(loss):
                logger.warning(f"step {step}: non-finite loss {loss.item()}; skipping the sample")
                continue
            loss.backward()
            loss_sum += float(loss.detach()) * cfg.gradient_accumulation_steps
            used += 1
        if used == 0:
            optimizer.zero_grad(set_to_none=True)
            continue
        if step == init_step + 1 or step % cfg.grad_check_interval == 0:
            for p in trainable:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"non-finite LoRA gradient at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if step % cfg.log_interval == 0:
            loss_v = loss_sum / used
            logger.info(f"Step {step}/{cfg.max_train_steps}: loss {loss_v:.4f} (mse {mse_sum / used:.4f} "
                        f"edge {edge_sum / used:.4f} masked {masked}/{used}) grad_norm {float(grad_norm):.4f} "
                        f"step_time {dt:.2f}s avg {sum(step_times) / len(step_times):.2f}s "
                        f"lr {lr_scheduler.get_last_lr()[0]:.2e}")
            tb.add_scalar("Loss/sft2", loss_v, step)
            tb.add_scalar("Loss/mse_weighted", mse_sum / used, step)
            tb.add_scalar("Loss/edge", edge_sum / used, step)
            tb.add_scalar("Grad/norm", float(grad_norm), step)
            tb.add_scalar("Time/step", dt, step)
        if step % cfg.checkpoint_interval == 0 or step == cfg.max_train_steps:
            save_checkpoint(model, ckpt_dir, step, global_rank, epoch=epoch, optimizer=optimizer,
                            scheduler=lr_scheduler, dataloader=loader)
        if step % cfg.gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
    tb.close()
    clean_dist_env()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Azyah 0811 LoRA (live 2-step rollout vs real targets)")
    ap.add_argument("--config", required=True)
    main(load_config_class_from_pyfile(ap.parse_args().config)())
