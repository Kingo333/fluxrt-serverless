"""The live model's own two-step render, written out so that gradients can flow through it (C2, 2026-11-21).

The kiosk runs the distilled FLUX.2-klein-4B for 2 steps from one fixed noise (one seed for every frame), with the
camera frame and the garment photo as reference pictures. Training so far changed the BASE model's single-step
prediction and hoped the change carries over; the scores say it does not. Here the same computation the live
worker does (Flux2KleinPipeline.__call__ at the pinned diffusers commit: guidance 1.0, no second pass) is split
into its parts, so a LoRA add-on, or the starting noise itself, can be trained on the picture the kiosk shows.

  prepare_references(pipe, [frame, garment])       -> reference tokens and their ids (as the pipeline makes them)
  start_noise(pipe, seed, height, width)           -> the fixed noise tokens of that seed, and their ids
  live_sigmas(pipe, tokens, steps)                 -> the noise levels the pipeline uses (2 steps: 1.0, ~0.88, 0)
  sample(transformer, noise, ...)                  -> the clean latent tokens after the steps (differentiable)
  to_latents(pipe, tokens, ids, height, width)     -> the latents exactly as pipe(..., output_type="latent")
  answer_tokens(pipe, picture)                     -> a real picture in the same token space (the training answer)

tests/training/test_live_sampler_torch.py checks, on a tiny pipeline, that sample() gives the pipeline's own
output for the same seed, to rounding.
"""

from __future__ import annotations

import numpy as np
import torch


def _preprocess(pipe, img):
    """One reference picture as the pipeline prepares it (cropped to a multiple of 16, at most 1024x1024)."""
    w, h = img.size
    if w * h > 1024 * 1024:
        img = pipe.image_processor._resize_to_target_area(img, 1024 * 1024)
        w, h = img.size
    m = pipe.vae_scale_factor * 2
    return pipe.image_processor.preprocess(img, height=(h // m) * m, width=(w // m) * m, resize_mode="crop")


@torch.no_grad()
def prepare_references(pipe, images, device=None, batch_size: int = 1):
    """(reference tokens (B, N, C), their ids (B, N, 4)) for the list of PIL pictures [camera frame, garment]."""
    device = device or pipe.transformer.device
    cond = [_preprocess(pipe, im) for im in images]
    return pipe.prepare_image_latents(images=cond, batch_size=batch_size, generator=None, device=device, dtype=pipe.vae.dtype)


def start_noise(pipe, seed: int, height: int, width: int, device=None, dtype=None, batch_size: int = 1):
    """(noise tokens (B, N, C), ids): the very noise pipe(..., generator=manual_seed(seed)) starts from."""
    device = device or pipe.transformer.device
    dtype = dtype or pipe.transformer.dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe.prepare_latents(batch_size=batch_size, num_latents_channels=pipe.transformer.config.in_channels // 4,
                                height=height, width=width, dtype=dtype, device=device, generator=gen)


def live_sigmas(pipe, n_tokens: int, steps: int = 2, device=None) -> torch.Tensor:
    """The noise levels of the pipeline's schedule for a picture of n_tokens tokens: steps + 1 values, the last 0."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    device = device or pipe.transformer.device
    sig = np.linspace(1.0, 1 / steps, steps)
    if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
        sig = None
    retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sig, mu=compute_empirical_mu(image_seq_len=n_tokens, num_steps=steps))
    return pipe.scheduler.sigmas.detach().clone().to(device=device, dtype=torch.float32)


def sample(transformer, noise, latent_ids, ref_tokens, ref_ids, prompt_embeds, text_ids, sigmas, keep_grad_through_steps: bool = True):
    """The clean latent tokens after the steps. noise: (B, N, C). Every step is the pipeline's: the model sees the
    current tokens followed by the reference tokens, and the tokens move by (next level - this level) x its answer.
    keep_grad_through_steps=False cuts the gradient between the steps (each step is then trained on its own)."""
    x = noise
    n = x.shape[1]
    ids = latent_ids if ref_tokens is None else torch.cat([latent_ids, ref_ids], dim=1)
    for i in range(len(sigmas) - 1):
        inp = x if ref_tokens is None else torch.cat([x, ref_tokens.to(x.dtype)], dim=1)
        t = sigmas[i].to(x.dtype).expand(x.shape[0])            # the pipeline passes timestep / 1000 = this level
        v = transformer(hidden_states=inp.to(transformer.dtype), timestep=t.to(transformer.dtype), guidance=None,
                        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=ids, return_dict=False)[0][:, :n]
        x = x + (sigmas[i + 1] - sigmas[i]).to(x.dtype) * v.to(x.dtype)
        if not keep_grad_through_steps and i < len(sigmas) - 2:
            x = x.detach()
    return x


def _bn(pipe, like):
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(like.device, like.dtype)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(like.device, like.dtype)
    return mean, std


def to_latents(pipe, tokens, latent_ids, height: int, width: int):
    """The latents as pipe(..., output_type="latent") returns them (un-packed, un-normalised, un-patchified)."""
    lh = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    lw = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    lat = pipe._unpack_latents_with_ids(tokens, latent_ids, lh // 2, lw // 2)
    mean, std = _bn(pipe, lat)
    return pipe._unpatchify_latents(lat * std + mean)


@torch.no_grad()
def answer_tokens(pipe, img, device=None):
    """A real picture as tokens in the space sample() works in (B=1, N, C): what the clean result should be."""
    device = device or pipe.transformer.device
    x = _preprocess(pipe, img).to(device=device, dtype=pipe.vae.dtype)
    return pipe._pack_latents(pipe._encode_vae_image(image=x, generator=None))
