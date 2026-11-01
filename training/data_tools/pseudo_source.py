"""Generate pseudo-sources for staged pairs (plan G1): the same person wearing a
*different* garment, made by editing the real target with the deployed JoyAI.

  # GPU Pod with the training volume (paid step, needs approval):
  python3 -m data_tools.pseudo_source STAGING_DIR --dit /path/joyai_video_edit_dit_0811.pth \
      --vae /path/vae --text-encoder /path/MiMo-VL-7B-RL-2508 [--limit N] [--workers-note]

For each staged pair without a source:
  1. pick a random plain garment for its category (seeded by the pair key, so reruns match)
  2. run a text-only JoyAI edit on the real target ("Substitute the current top with a plain
     navy crew-neck t-shirt ...") -- no reference image
  3. keep it only if the garment area changed and the rest did not (quality filter)
  4. write source.png / source.mp4 at the target's size, and record what was done in meta.json

Stills are pushed as a short static clip and the last output frame is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .staging import KEEP, find, iter_pairs

COLOURS = ["navy", "black", "white", "grey", "olive green", "burgundy", "beige", "sky blue",
           "mustard yellow", "forest green", "charcoal", "cream", "coral", "teal"]
GARMENTS = {
    "upper": ["crew-neck t-shirt", "long-sleeve t-shirt", "button-up shirt", "hoodie",
              "knit sweater", "polo shirt", "tank top"],
    "lower": ["jeans", "chino trousers", "joggers", "pleated skirt", "shorts"],
    "dress": ["midi dress", "shirt dress", "slip dress"],
}
TARGET_WORD = {"upper": "top", "lower": "pants or skirt", "dress": "outfit"}


def edit_prompt(key: str, category: str) -> str:
    rng = random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
    colour, garment = rng.choice(COLOURS), rng.choice(GARMENTS[category])
    return f"Substitute the current {TARGET_WORD[category]} with a plain {colour} {garment}. {KEEP}"


class Editor(Protocol):
    name: str

    def edit(self, frames: list, prompt: str, size: tuple[int, int]) -> list:
        """frames: PIL images; returns edited PIL images (same count) at `size`."""


@dataclass
class FilterResult:
    ok: bool
    inside_change: float
    outside_change: float
    reason: str = ""


def quality_filter(before, after, mask=None, min_inside=0.06, max_outside=0.05, min_global=0.02, max_global=0.35) -> FilterResult:
    """Mean absolute pixel change (0..1). With a mask (white = garment area): the garment
    must change and the rest must not. Without one: the edit must be visible but partial."""
    import numpy as np
    a = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(after.convert("RGB").resize(before.size), dtype=np.float32) / 255.0
    diff = np.abs(a - b).mean(axis=2)
    if mask is not None:
        m = np.asarray(mask.convert("L").resize(before.size), dtype=np.float32) / 255.0 > 0.5
        if m.sum() == 0 or (~m).sum() == 0:
            return FilterResult(False, 0.0, 0.0, "empty mask")
        inside, outside = float(diff[m].mean()), float(diff[~m].mean())
        if inside < min_inside:
            return FilterResult(False, inside, outside, "garment barely changed")
        if outside > max_outside:
            return FilterResult(False, inside, outside, "face/background changed too much")
        return FilterResult(True, inside, outside)
    g = float(diff.mean())
    if g < min_global:
        return FilterResult(False, g, g, "barely changed")
    if g > max_global:
        return FilterResult(False, g, g, "changed too much")
    return FilterResult(True, g, g)


ALIGN = 24  # JoyAI snaps sizes to vae.stem.stride * 8 (= 24; live sessions are 480x840) and trains with spatial_multiple=24


def _session_size(w: int, h: int, long_side: int = 840) -> tuple[int, int]:
    """JoyAI session size with the target's aspect ratio, sides multiples of 24."""
    s = long_side / max(w, h)
    return max(ALIGN, round(w * s / ALIGN) * ALIGN), max(ALIGN, round(h * s / ALIGN) * ALIGN)


def _read_video(path: Path):
    import av
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        fps = float(stream.average_rate or 24)
        frames = [f.to_image() for f in c.decode(stream)]
    return frames, fps


def _write_video(path: Path, frames, fps: float) -> None:
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=round(fps))
        s.width, s.height = frames[0].size
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "18"}
        for im in frames:
            for pkt in s.encode(av.VideoFrame.from_image(im.convert("RGB"))):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def process(staging_dir: str, editor: Editor, limit: int | None = None, still_frames: int = 9) -> dict:
    from PIL import Image
    stats = {"done": 0, "rejected": 0, "skipped_has_source": 0, "reasons": {}}
    for d, meta in iter_pairs(Path(staging_dir)):
        if find(d, "source"):
            stats["skipped_has_source"] += 1
            continue
        if limit is not None and stats["done"] + stats["rejected"] >= limit:
            break
        tgt = find(d, "target")
        prompt = edit_prompt(d.name, meta["category"])
        mask_path = find(d, "mask")
        mask = Image.open(mask_path) if mask_path else None
        t0 = time.time()
        if tgt.suffix == ".mp4":
            frames, fps = _read_video(tgt)
            size = frames[0].size
            out = editor.edit(frames, prompt, _session_size(*size))
            out = [im.resize(size) for im in out]
            check = quality_filter(frames[len(frames) // 2], out[len(out) // 2], mask)
            if check.ok:
                _write_video(d / "source.mp4", out, fps)
        else:
            target = Image.open(tgt).convert("RGB")
            out = editor.edit([target] * still_frames, prompt, _session_size(*target.size))
            src = out[-1].resize(target.size)
            check = quality_filter(target, src, mask)
            if check.ok:
                buf = io.BytesIO()
                src.save(buf, format="PNG")
                (d / "source.png").write_bytes(buf.getvalue())
        meta["pseudo_source"] = {"model": editor.name, "prompt": prompt, "ok": check.ok,
                                 "inside_change": round(check.inside_change, 4),
                                 "outside_change": round(check.outside_change, 4),
                                 "reason": check.reason, "seconds": round(time.time() - t0, 2)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        if check.ok:
            stats["done"] += 1
        else:
            stats["rejected"] += 1
            stats["reasons"][check.reason] = stats["reasons"].get(check.reason, 0) + 1
    return stats


class JoyAIEditor:
    """Text-only edits with the deployed JoyAI streaming runtime (GPU)."""

    name = "joyai-0811"

    def __init__(self, dit: str, vae: str, text_encoder: str, device: str = "cuda:0", steps: int = 2):
        import sys
        import os
        deploy = os.environ.get("JOYAI_DEPLOY_DIR", "/workspace/azyah-vton/JoyAI-Video-Edit/deploy")
        sys.path.insert(0, deploy)
        from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings  # noqa: E402
        self._settings_cls = StreamingSettings
        self.steps = steps
        # Give each VAE role its own instance (shared weights), like the server does
        # (start_upstream.sh passes --vae-encode/decode/pseudo-device). Without these the
        # encode worker and the decode/pseudo-encode worker share ONE VAE object and its
        # causal feature cache across threads; on moving video one thread clears the cache
        # mid-encode -> "Input temporal dimension is expected to be 1 when cache_x is None".
        self.rt = JoyOmniRuntime.load(dit, vae_ckpt=vae, text_encoder_ckpt=text_encoder, device=device,
                                      vae_encode_device=device, vae_decode_device=device,
                                      vae_pseudo_device=device, postprocess_device=device,
                                      warmup_height=840, warmup_width=624)

    def edit(self, frames, prompt, size):
        w, h = size
        settings = self._settings_cls(height=h, width=w, num_inference_steps=self.steps, max_temporal_ids=8)
        session = self.rt.create_v2v_session(prompt=prompt, settings=settings)
        jpegs: dict[int, bytes] = {}
        try:
            for i, f in enumerate(frames):
                for r in session.push_frame(f, frame_meta={"seq": i, "t_capture_ms": 0.0}):
                    self._collect(r, jpegs)
            session.flush_pending()
            deadline = time.time() + 120
            while len(jpegs) < len(frames) and time.time() < deadline:
                r = session.wait_async_result(timeout=0.5)
                if r is not None:
                    self._collect(r, jpegs)
        finally:
            session.close(timeout=60)
        from PIL import Image
        last = None
        out = []
        for i in range(len(frames)):
            data = jpegs.get(i)
            im = Image.open(io.BytesIO(data)).convert("RGB") if data else last
            if im is None:
                raise RuntimeError("JoyAI returned no frames")
            out.append(im)
            last = im
        return out

    @staticmethod
    def _collect(result, jpegs):
        for meta, data in zip(result.source_metas, result.jpegs or []):
            jpegs[meta.get("seq")] = data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging_dir")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    editor = JoyAIEditor(a.dit, a.vae, a.text_encoder)
    print(json.dumps(process(a.staging_dir, editor, a.limit), indent=2))


if __name__ == "__main__":
    main()
