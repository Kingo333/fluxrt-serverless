"""Rebuild training inputs the LIVE way (step 3a): the same person in a *different* real
garment, made by the live JoyAI clone (0811) with a FIXED garment photo as reference, the
way the app runs it, instead of P8's text-only edits.

  # GPU Pod with the training volume (paid step, owner's yes first):
  python3 -m data_tools.live_source STAGING_DIR [STAGING_DIR ...] --dit /workspace/.../joyai_video_edit_dit_0811.pth \
      --vae /workspace/azyah-vton/checkpoints/vae --text-encoder /workspace/azyah-vton/checkpoints/MiMo-VL-7B-RL-2508 \
      [--remove-frac 0.25] [--limit N] [--videos --categories dress --max-frames 97]

Why: the owner checked the P8 pseudo-source VIDEOS (2026-11-04): with no garment photo the
clothes drift from chunk to chunk. A fixed reference photo keeps the edit steady, and it is the
app's own input format (camera frame + product photo + instruction).

For each staged pair (resumable: pairs with meta["live_source"] are skipped):
  1. reference = the garment photo of ANOTHER staged pair of the same category, chosen with a
     seed from the pair key (reruns pick the same one)
  2. "remove" augmentation (step 3a; owner saw a leftover belt and a doubled bag): a share
     (--remove-frac, 0.25) also gets an extra item in the prompt that the real target does not
     have (belt, handbag, scarf, hat; by category). Training on extra-item input -> real target
     teaches the LoRA to drop what is not in the garment photo and never to add objects.
  3. the live 0811 session edits the real target with that reference + the app-style
     instruction (data_tools.staging.instruction), stills as a short static clip (last frame kept)
  4. quality filter (pseudo_source.quality_filter): the garment must change and the rest must
     not; extra-item edits may change a little more outside the garment (the added item)
  5. writes live_source.png / live_source.mp4 next to the pair. The old source.* is NEVER
     touched, so earlier runs stay reproducible; pack.py --prefer-live-source uses the new one.
Videos (ViViD) only with --videos (owner opt-in; ViViD is research-only).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path

from .pseudo_source import JoyAIEditor, _read_video, _session_size, _write_video, quality_filter
from .staging import KEEP, find, instruction, iter_pairs

EXTRA_ITEMS = {
    "upper": ["a black leather belt", "a small handbag on one shoulder", "a patterned silk scarf", "a baseball cap"],
    "lower": ["a brown leather belt", "a small crossbody bag", "a knit beanie", "a black leather belt"],
    "dress": ["a thin waist belt", "a handbag held in one hand", "a straw sun hat", "a patterned scarf"],
}
MAX_OUTSIDE = 0.05        # plain edits: face/background must stay (as pseudo_source)
MAX_OUTSIDE_EXTRA = 0.12  # extra-item edits add something outside the garment area


def _seed(key: str, salt: str) -> int:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16)


def choose_reference(key: str, category: str, pool: dict[str, list[tuple[str, Path]]]) -> tuple[str, Path] | None:
    """(ref_key, garment photo) of another pair of the same category, fixed per key."""
    cands = [c for c in pool.get(category, []) if c[0] != key]
    if not cands:
        return None
    return random.Random(_seed(key, "ref")).choice(sorted(cands))


def remove_extra(key: str, category: str, frac: float) -> str | None:
    """The extra item for a 'remove' example, or None (about `frac` of keys get one)."""
    if frac <= 0:
        return None
    if _seed(key, "remove") / 0xFFFFFFFF >= frac:
        return None
    return random.Random(_seed(key, "item")).choice(EXTRA_ITEMS.get(category, EXTRA_ITEMS["upper"]))


def live_prompt(category: str, extra: str | None = None) -> str:
    """The app's instruction (exact twin) with the extra item inserted before the keep sentence."""
    text = instruction(category)
    if not extra:
        return text
    assert text.endswith(KEEP)
    return f"{text[: -len(KEEP)]}Also add {extra}. {KEEP}"


def build_pool(staging_dirs: list[str]) -> dict[str, list[tuple[str, Path]]]:
    pool: dict[str, list[tuple[str, Path]]] = {}
    for sd in staging_dirs:
        for d, meta in iter_pairs(Path(sd)):
            ref = find(d, "ref")
            if ref is not None and meta.get("split", "train") == "train":
                pool.setdefault(meta["category"], []).append((d.name, ref))
    return pool


def process(staging_dirs: list[str], editor, remove_frac: float = 0.25, limit: int | None = None,
            still_frames: int = 9, videos: bool = False, categories: set[str] | None = None,
            max_frames: int | None = None) -> dict:
    from PIL import Image
    pool = build_pool(staging_dirs)
    stats = {"done": 0, "rejected": 0, "skipped": 0, "with_extra_item": 0, "reasons": {}}
    for sd in staging_dirs:
        for d, meta in iter_pairs(Path(sd)):
            if limit is not None and stats["done"] + stats["rejected"] >= limit:
                return stats
            tgt = find(d, "target")
            if meta.get("live_source") or tgt is None or (categories and meta.get("category") not in categories):
                stats["skipped"] += 1
                continue
            if tgt.suffix == ".mp4" and not videos:
                stats["skipped"] += 1
                continue
            cat = meta["category"]
            ref = choose_reference(d.name, cat, pool)
            if ref is None:
                stats["skipped"] += 1
                continue
            extra = remove_extra(d.name, cat, remove_frac)
            prompt = live_prompt(cat, extra)
            ref_img = Image.open(ref[1]).convert("RGB")
            mask_path = find(d, "mask")
            mask = Image.open(mask_path) if mask_path else None
            max_out = MAX_OUTSIDE_EXTRA if extra else MAX_OUTSIDE
            t0 = time.time()
            if tgt.suffix == ".mp4":
                frames, fps = _read_video(tgt)
                frames = frames[:max_frames] if max_frames else frames
                size = frames[0].size
                out = [im.resize(size) for im in editor.edit(frames, prompt, _session_size(*size), ref_image=ref_img)]
                check = quality_filter(frames[len(frames) // 2], out[len(out) // 2], mask, max_outside=max_out)
                if check.ok:
                    _write_video(d / "live_source.mp4", out, fps)
            else:
                target = Image.open(tgt).convert("RGB")
                out = editor.edit([target] * still_frames, prompt, _session_size(*target.size), ref_image=ref_img)
                src = out[-1].resize(target.size)
                check = quality_filter(target, src, mask, max_outside=max_out)
                if check.ok:
                    buf = io.BytesIO()
                    src.save(buf, format="PNG")
                    (d / "live_source.png").write_bytes(buf.getvalue())
            meta["live_source"] = {"model": editor.name, "ref_key": ref[0], "prompt": prompt, "extra_item": extra,
                                   "ok": check.ok, "inside_change": round(check.inside_change, 4),
                                   "outside_change": round(check.outside_change, 4), "reason": check.reason,
                                   "seconds": round(time.time() - t0, 2)}
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            if check.ok:
                stats["done"] += 1
                stats["with_extra_item"] += int(bool(extra))
            else:
                stats["rejected"] += 1
                stats["reasons"][check.reason] = stats["reasons"].get(check.reason, 0) + 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("staging_dirs", nargs="+")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--remove-frac", type=float, default=0.25)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--videos", action="store_true", help="also clip targets (ViViD; owner opt-in, research-only)")
    ap.add_argument("--categories", nargs="*", help="only these categories (e.g. dress)")
    ap.add_argument("--max-frames", type=int, help="clips: edit only the first N frames (97 = 1 + 12 chunks)")
    a = ap.parse_args()
    editor = JoyAIEditor(a.dit, a.vae, a.text_encoder)
    stats = process(a.staging_dirs, editor, a.remove_frac, a.limit, videos=a.videos,
                    categories=set(a.categories) if a.categories else None, max_frames=a.max_frames)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
