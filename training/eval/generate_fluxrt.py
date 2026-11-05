"""FluxRT eval: quality AND speed, base vs garment LoRA, on the same held-out pairs as P9e.

  python -m eval.generate_fluxrt --model-dir $AZ_CKPT/FLUX.2-klein-4B [--lora DIR/pytorch_lora_weights.safetensors] \\
      --name base|lora --staging /workspace/azyah-vton/data/staging/vivid --out /workspace/azyah-vton/eval/fluxrt --max-items 16

FluxRT edits each camera frame with FLUX.2-klein-4B: image=[camera_frame, garment], 2 steps,
guidance 1, at the endpoint size (288x512; garment white-padded to 384x512, as server.py does).
Here every frame of the held-out pseudo-source clip (first --frames) goes through the same
pipeline call, so the clip can be scored by eval/run_eval.py (garment colour/detail,
preservation, flicker) next to P9e's JoyAI results.

Speed: per-frame latency with CUDA sync (first 3 frames are warm-up and not counted),
written to perf_<name>.json (fps, latency_ms_p50/p95, gpu_mem_gb). This is the model's own
cost per frame without the live engine's extras (spatial cache, RIFE interpolation, Flow
Upscaler), so live fps on the endpoint is higher; compare base vs LoRA here, live via
window.__azyahLiveCam.stats() on the phone.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.generate_vivid import cross_pairs, test_items  # noqa: E402


def _first(d: Path, stem: str) -> Path:
    return next(p for p in sorted(d.iterdir()) if p.stem == stem and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))


def pair_items(root: Path, max_items: int) -> list[Path]:
    """--pairs: regression items, each <item>/person.*, ref.*, meta.json (no ground truth)."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            _first(d, "person"), _first(d, "ref")
        except StopIteration:
            continue
        if (d / "meta.json").exists():
            out.append(d)
        if len(out) >= max_items:
            break
    return out


def item_zone_mask(category, w: int, h: int):
    """Where the item should change, for items without a garment mask: the bottom quarter for shoes
    (full-body photo, feet at the bottom). run_eval then scores the shoe colour/detail inside it
    and the person (preservation) outside it. None for other categories (whole frame)."""
    from PIL import Image
    if str(category or "").lower() not in ("shoes", "footwear"):
        return None
    m = Image.new("L", (w, h), 0)
    m.paste(255, (0, int(h * 0.75), w, h))
    return m


def still_items(staging: Path, max_items: int) -> list[Path]:
    """--stills: held-out still pairs (Glance) with a steady pseudo-source photo and a real target
    photo. FluxRT edits one frame at a time, so a still with ground truth is its fairest test."""
    out = []
    for d in sorted(p for p in staging.iterdir() if p.is_dir()):
        mp = d / "meta.json"
        if not mp.exists() or not (d / "source.png").exists():
            continue
        if not any((d / f"target.{e}").exists() for e in ("png", "jpg", "jpeg")):
            continue
        meta = json.loads(mp.read_text())
        if meta.get("split") == "test" and (meta.get("pseudo_source") or {}).get("ok", True):
            out.append(d)
        if len(out) >= max_items:
            break
    return out


def summarize(latencies_s: list[float], gpu_mem_gb: float | None) -> dict:
    ms = sorted(x * 1000 for x in latencies_s)
    if not ms:
        return {"fps": None, "latency_ms_p50": None, "latency_ms_p95": None, "gpu_mem_gb": gpu_mem_gb, "frames": 0}
    p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    mean = statistics.fmean(ms)
    return {"fps": round(1000 / mean, 2), "latency_ms_p50": round(statistics.median(ms), 1),
            "latency_ms_p95": round(p95, 1), "gpu_mem_gb": gpu_mem_gb, "frames": len(ms)}


def _cover(img, w, h):
    from PIL import Image
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _pad(img, w, h):
    from PIL import Image
    s = min(w / img.width, h / img.height)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def _read_frames(path: Path, n: int):
    import av
    with av.open(str(path)) as c:
        out = []
        for f in c.decode(video=0):
            out.append(f.to_image().convert("RGB"))
            if len(out) >= n:
                break
    return out


def _write_mp4(path: Path, frames, fps: int = 24) -> None:
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        s.options = {"crf": "18"}
        for im in frames:
            for pkt in s.encode(av.VideoFrame.from_image(im)):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--name", required=True)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=16)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--size", default="288x512")
    ap.add_argument("--ref-size", default="384x512")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--cross", action="store_true",
                    help="clips: input = the real clip (target.mp4) with another item's garment (steady input)")
    ap.add_argument("--stills", action="store_true",
                    help="still pairs (e.g. staging/glance_upper): pseudo-source photo in, real photo as target")
    ap.add_argument("--pairs", action="store_true",
                    help="regression items (e.g. the shoe set from eval.make_shoe_set): <item>/person.*, ref.*, "
                         "meta.json {category, name, description}; no target. Prompt = the app's buildTryOnPrompt")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength (fallback test, e.g. 0.5)")
    a = ap.parse_args(argv)

    import shutil
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image
    from data_tools.fluxrt_prompt import fluxrt_prompt

    w, h = (int(v) for v in a.size.lower().split("x"))
    rw, rh = (int(v) for v in a.ref_size.lower().split("x"))
    pipe = Flux2KleinPipeline.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    if a.lora:
        pipe.load_lora_weights(a.lora, adapter_name="garment")
        if a.lora_scale != 1.0:
            pipe.set_adapters(["garment"], adapter_weights=[a.lora_scale])
        print(f"LoRA {a.lora} at scale {a.lora_scale}", flush=True)
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.reset_peak_memory_stats()
    out_root = Path(a.out)
    latencies: list[float] = []
    if a.pairs:
        pairs = [(d, d) for d in pair_items(Path(a.staging), a.max_items)]
    elif a.stills:
        pairs = [(d, d) for d in still_items(Path(a.staging), a.max_items)]
    elif a.cross:
        pairs = cross_pairs(test_items(Path(a.staging), max(a.max_items * 4, 16)))[:a.max_items]
    else:
        pairs = [(d, d) for d in test_items(Path(a.staging), a.max_items)]
    ext = "png" if (a.stills or a.pairs) else "mp4"
    for d, gd in pairs:
        meta = json.loads((d / "meta.json").read_text())
        dest = out_root / d.name
        result = dest / f"out_{a.name}.{ext}"
        if result.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if a.pairs:
            frames = [_cover(Image.open(_first(d, "person")).convert("RGB"), w, h)] * 4
        elif a.stills:
            frames = [_cover(Image.open(d / "source.png").convert("RGB"), w, h)] * 4  # 3 warm-up + 1 kept
        else:
            frames = [_cover(f, w, h) for f in _read_frames(d / ("target.mp4" if a.cross else "source.mp4"), a.frames)]
        ref_path = next(p for p in gd.iterdir() if p.stem == "ref")
        garment = _pad(Image.open(ref_path).convert("RGB"), rw, rh)
        prompt = fluxrt_prompt(meta.get("category"), meta.get("name"), meta.get("description"))
        outs = []
        for i, frame in enumerate(frames):
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = pipe(prompt=prompt, image=[frame, garment], height=h, width=w, guidance_scale=1.0,
                       num_inference_steps=a.steps, generator=gen).images[0]
            torch.cuda.synchronize()
            if i >= 3:
                latencies.append(time.perf_counter() - t0)
            outs.append(img.convert("RGB").resize((w, h)))
        if a.stills or a.pairs:
            outs[-1].save(result)
        else:
            _write_mp4(result, outs)
        if not (dest / f"input.{ext}").exists():
            if a.pairs:
                frames[0].save(dest / "input.png")
                zone = item_zone_mask(meta.get("category"), w, h)
                if zone is not None:
                    zone.save(dest / "mask.png")
                (dest / "item.json").write_text(json.dumps(meta))
            elif a.stills:
                frames[0].save(dest / "input.png")
                tgt = next(d / f"target.{e}" for e in ("png", "jpg", "jpeg") if (d / f"target.{e}").exists())
                _cover(Image.open(tgt).convert("RGB"), w, h).save(dest / "target.png")
                if (d / "mask.png").exists():  # Glance garment mask: lets run_eval score the person outside it
                    _cover(Image.open(d / "mask.png").convert("L"), w, h).save(dest / "mask.png")
            else:
                _write_mp4(dest / "input.mp4", frames)
                if a.cross:
                    (dest / "cross.json").write_text(json.dumps({"person": d.name, "garment": gd.name}))
                else:
                    shutil.copy(d / "target.mp4", dest / "target.mp4")
            shutil.copy(ref_path, dest / f"ref{ref_path.suffix}")
        perf = summarize(latencies, round(torch.cuda.max_memory_reserved() / 1e9, 2))
        (dest / f"perf_{a.name}.json").write_text(json.dumps(perf))
        print(json.dumps({"key": d.name, "model": a.name, **perf}), flush=True)

if __name__ == "__main__":
    main()
