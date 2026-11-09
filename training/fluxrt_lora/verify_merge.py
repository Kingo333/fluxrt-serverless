"""Prove a merged LoRA file equals the live blend it was made from (GPU; a few minutes).

  python -m fluxrt_lora.verify_merge --model-dir CKPT/FLUX.2-klein-4B --inputs A.safetensors B.safetensors \
      --weights 0.5 0.5 --scale 0.65 --merged OUT/pytorch_lora_weights.safetensors --clip-list clips.json [--items 2]

Renders the same frames twice with the same seed: once with the two LoRAs loaded as separate adapters at
weights (w1*s, w2*s), once with the single merged file at strength 1.0. Prints the mean and max pixel
difference (0-255) per item. Identical maths gives differences near 0 (bf16 rounding: well under 1).
Exits 1 if any mean difference is above --tolerance (default 1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def render(pipe, items, frames_per_item, w, h, rw, rh, seed):
    import torch
    from PIL import Image
    from data_tools.fluxrt_prompt import compact_prompt
    from eval.generate_fluxrt import _cover, _pad, _read_frames
    outs = []
    for it in items:
        frames = [_cover(f, w, h) for f in _read_frames(Path(it["video"]), frames_per_item)]
        garment = _pad(Image.open(it["ref"]).convert("RGB"), rw, rh)
        prompt = it.get("prompt") or compact_prompt(it.get("category"), it.get("name"), it.get("description"))
        for frame in frames:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(prompt=prompt, image=[frame, garment], height=h, width=w, guidance_scale=1.0,
                       num_inference_steps=2, generator=gen).images[0]
            outs.append(img.convert("RGB"))
    return outs


def main(argv=None) -> dict:
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--inputs", nargs=2, required=True)
    ap.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--clip-list", required=True)
    ap.add_argument("--items", type=int, default=2)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", default="288x512")
    ap.add_argument("--ref-size", default="288x512")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--tolerance", type=float, default=1.0)
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    rw, rh = (int(v) for v in a.ref_size.lower().split("x"))
    items = json.loads(Path(a.clip_list).read_text())[: a.items]

    pipe = Flux2KleinPipeline.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights(a.inputs[0], adapter_name="first")
    pipe.load_lora_weights(a.inputs[1], adapter_name="second")
    pipe.set_adapters(["first", "second"], adapter_weights=[a.weights[0] * a.scale, a.weights[1] * a.scale])
    ref = render(pipe, items, a.frames, w, h, rw, rh, a.seed)
    pipe.unload_lora_weights()

    pipe.load_lora_weights(a.merged, adapter_name="merged")
    pipe.set_adapters(["merged"], adapter_weights=[1.0])
    got = render(pipe, items, a.frames, w, h, rw, rh, a.seed)

    report, worst = [], 0.0
    for k, (x, y) in enumerate(zip(ref, got)):
        d = np.abs(np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32))
        report.append({"frame": k, "mean_diff": round(float(d.mean()), 3), "max_diff": round(float(d.max()), 1)})
        worst = max(worst, float(d.mean()))
    ok = worst <= a.tolerance
    print(json.dumps({"ok": ok, "worst_mean_diff": round(worst, 3), "frames": report}, indent=1))
    if not ok:
        raise SystemExit(1)
    return {"ok": ok, "worst_mean_diff": worst}


if __name__ == "__main__":
    main()
