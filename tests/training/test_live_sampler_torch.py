"""The live two-step render written out for training (fluxrt_lora.live_sampler) must BE the pipeline's render: on a
tiny pipeline, the same seed gives the same latents as Flux2KleinPipeline.__call__, and gradients reach a LoRA and
the starting noise. Needs torch, peft and diffusers at the trainer's pinned commit (skipped without them, e.g. in CI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
diffusers = pytest.importorskip("diffusers")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
W, H = 64, 96            # the picture; the garment photo is another size, as live (288x512 and 528x704)


def tiny_pipe(seed=0):
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
    torch.manual_seed(seed)
    tr = Flux2Transformer2DModel(patch_size=1, in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16,
                                 num_attention_heads=2, joint_attention_dim=32, timestep_guidance_channels=16,
                                 mlp_ratio=2.0, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).eval()
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 16, 16), layers_per_block=1, latent_channels=2, norm_num_groups=4,
                             sample_size=64).eval()
    with torch.no_grad():                                   # statistics that are not the neutral 0 and 1
        vae.bn.running_mean.copy_(torch.linspace(-0.3, 0.4, vae.bn.running_mean.numel()))
        vae.bn.running_var.copy_(torch.linspace(0.5, 1.5, vae.bn.running_var.numel()))
    sch = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    return Flux2KleinPipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr, is_distilled=True)


def pictures(seed=1):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(seed)
    frame = Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), "RGB")
    garment = Image.fromarray(rng.integers(0, 255, (96, 80, 3), dtype=np.uint8), "RGB")
    return frame, garment


def text(seed=2, tokens=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, tokens, 32, generator=g)


def test_the_written_out_render_is_the_pipelines_own(tmp_path):
    from fluxrt_lora import live_sampler as ls
    pipe = tiny_pipe()
    pipe.set_progress_bar_config(disable=True)
    frame, garment = pictures()
    emb = text()
    for seed in (52, 54):
        want = pipe(prompt=None, prompt_embeds=emb, image=[frame, garment], height=H, width=W, guidance_scale=1.0,
                    num_inference_steps=2, generator=torch.Generator().manual_seed(seed), output_type="latent").images
        ref, ref_ids = ls.prepare_references(pipe, [frame, garment])
        noise, ids = ls.start_noise(pipe, seed, H, W)
        sig = ls.live_sigmas(pipe, noise.shape[1], 2)
        assert sig.shape == (3,) and float(sig[0]) == pytest.approx(1.0) and float(sig[-1]) == 0.0 and 0.5 < float(sig[1]) < 1.0
        emb2, text_ids = pipe.encode_prompt(prompt=None, prompt_embeds=emb)
        with torch.no_grad():
            got = ls.to_latents(pipe, ls.sample(pipe.transformer, noise, ids, ref, ref_ids, emb2, text_ids, sig), ids, H, W)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4), float((got - want).abs().max())
    # another seed is another picture (the check above is not trivially true)
    a = pipe(prompt=None, prompt_embeds=emb, image=[frame, garment], height=H, width=W, guidance_scale=1.0,
             num_inference_steps=2, generator=torch.Generator().manual_seed(52), output_type="latent").images
    assert float((a - want).abs().max()) > 1e-3


def test_the_live_noise_levels_at_the_kiosks_size_are_one_and_about_088():
    from fluxrt_lora import live_sampler as ls
    pipe = tiny_pipe()
    sig = ls.live_sigmas(pipe, 576, 2)                       # 288x512: 18 x 32 tokens
    assert [round(float(x), 2) for x in sig] == [1.0, 0.88, 0.0]


def test_gradients_reach_an_add_on_and_the_starting_noise():
    from peft import LoraConfig
    from fluxrt_lora import live_sampler as ls
    pipe = tiny_pipe()
    frame, garment = pictures()
    emb, text_ids = pipe.encode_prompt(prompt=None, prompt_embeds=text())
    ref, ref_ids = ls.prepare_references(pipe, [frame, garment])
    noise, ids = ls.start_noise(pipe, 52, H, W)
    sig = ls.live_sigmas(pipe, noise.shape[1], 2)
    answer = ls.answer_tokens(pipe, frame)
    assert answer.shape == noise.shape
    pipe.transformer.requires_grad_(False)
    pipe.transformer.add_adapter(LoraConfig(r=2, lora_alpha=2, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v"]))
    lora = [p for n, p in pipe.transformer.named_parameters() if "lora" in n]
    assert lora and all(p.requires_grad for p in lora)
    z = noise.clone().requires_grad_(True)
    out = ls.sample(pipe.transformer, z, ids, ref, ref_ids, emb, text_ids, sig)
    ((out - answer) ** 2).mean().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
    got = [p for n, p in pipe.transformer.named_parameters() if "lora_B" in n and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert got                                                # lora_A gets none at the zero start: B is zero
    # cutting the gradient between the steps still trains, and the starting noise then only feels the first step
    z2 = noise.clone().requires_grad_(True)
    out2 = ls.sample(pipe.transformer, z2, ids, ref, ref_ids, emb, text_ids, sig, keep_grad_through_steps=False)
    assert torch.allclose(out2, out.detach(), atol=1e-6)
    ((out2 - answer) ** 2).mean().backward()
    assert z2.grad is None or float(z2.grad.abs().sum()) == 0.0


def test_a_few_steps_on_the_starting_noise_bring_the_render_closer_to_the_answer():
    from fluxrt_lora import live_sampler as ls
    pipe = tiny_pipe()
    pipe.transformer.requires_grad_(False)
    frame, garment = pictures()
    emb, text_ids = pipe.encode_prompt(prompt=None, prompt_embeds=text())
    ref, ref_ids = ls.prepare_references(pipe, [frame, garment])
    noise, ids = ls.start_noise(pipe, 52, H, W)
    sig = ls.live_sigmas(pipe, noise.shape[1], 2)
    answer = ls.answer_tokens(pipe, frame)
    z = noise.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = ((ls.sample(pipe.transformer, z, ids, ref, ref_ids, emb, text_ids, sig) - answer) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.9
