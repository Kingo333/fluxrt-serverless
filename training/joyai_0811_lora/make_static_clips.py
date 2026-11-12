#!/usr/bin/env python3
"""Steady training clips for the 0811 LoRA: each r2i still triple (garment photo, person
pseudo-source still, real target still) becomes an iv2v sample whose two videos are the
same still repeated, at the live 480x840 portrait size and 24 fps.

  python3 make_static_clips.py --manifest-dir $AZ_MANIFEST_DIR --out DIR [--frames 25] [--max 3000]
         [--mode research|commercial] [--vivid-videos] [--staging-dirs DIR ...] [--detail-oversample 2]

Why stills: the owner checked the P8 ViViD pseudo-source VIDEOS (2026-11-04) and the clothes
change within every clip, so they would teach the live model to flicker. A still pushed as a
short static clip (as data_tools/pseudo_source.py did for stills) is steady by construction.

- reads <manifest-dir>/i2i.json (r2i shards: 0 garment, 1 person source, 2 real target)
- person and target get the same cover crop (the person still is first resized to the
  target's size), so the only difference between the two clips is the garment
- captions: the shard's caption (short live instruction) and caption_detailed (the full
  instruction with name/description, the form the app sends); both are data_tools.staging
  .instruction(), the exact twin of the app's buildTryOnInstruction()
- --frames must be 8n+1 (the VAE's time factor); 25 = 4 latent chunks = the live sink chunk
  plus the 3-chunk window, so the trained rollout sees the same history layout as live
- train split only; --mode commercial drops research-only sources; sources that need consent
  must carry a consent_id (Sample.check)
- --vivid-videos (OFF by default): also copy the ViViD iv2v clip pairs from v2v.json as they
  are. They flicker (see above); only for an explicit, owner-approved experiment.

- --staging-dirs (step 3a): the staged pair folders behind the manifest (glance_upper,
  glance_lower, ...). Each sample's mask.png (Glance agnostic mask, white = garment) gets the
  same cover crop as the clips and is written to DIR/masks/<sample key>.png; train.py weights
  the loss inside it (AZ_MASK_WEIGHT) and adds the edge term there (AZ_EDGE_WEIGHT).
- --detail-oversample N (step 3a, default 2; 1 = off): every garment photo gets a detail score
  (eval.metrics edge energy on its non-white pixels: prints, trims, ribs, ruffles); the top
  third is repeated N times (keys <key>_d2, ...) so detailed garments get more of the training.

Writes DIR/shards/*.tar, DIR/manifest/v2v.json + selection.json (provenance travels with the
LoRA: commercial_use_cleared, trained_with_vivid), DIR/masks/ and DIR/build_summary.json.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools import manifests, sources  # noqa: E402
from eval import metrics  # noqa: E402
from data_tools.shards import Sample, write_shards  # noqa: E402

LIVE_SIZE = (480, 840)  # width, height: the live JOYAI_SESSION_WIDTH x JOYAI_SESSION_HEIGHT
LIVE_FPS = 24


def read_shard(path: str) -> dict[str, dict[str, bytes]]:
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path) as tar:
        for m in tar:
            if m.isfile():
                key, _, name = m.name.partition(".")
                groups.setdefault(key, {})[name] = tar.extractfile(m).read()
    return groups


def iter_manifest(manifest_dir: Path, group: str):
    path = manifest_dir / f"{group}.json"
    if not path.exists():
        return
    for shard in json.loads(path.read_text())["resolved_files"]:
        for key, files in read_shard(shard).items():
            yield key, files, json.loads(files.get("json", b"{}"))


def _pick(files: dict[str, bytes], n: str) -> bytes | None:
    return files.get(f"{n}.png") or files.get(f"{n}.jpg")


def cover_box(src_w: int, src_h: int, w: int, h: int) -> tuple[float, tuple[int, int, int, int]]:
    """Scale and crop box that fill w x h from src_w x src_h, centred (no distortion)."""
    scale = max(w / src_w, h / src_h)
    rw, rh = max(w, round(src_w * scale)), max(h, round(src_h * scale))
    left, top = (rw - w) // 2, (rh - h) // 2
    return scale, (left, top, left + w, top + h)


def cover(img, w: int, h: int):
    from PIL import Image
    scale, box = cover_box(img.width, img.height, w, h)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    return img.crop(box)


def static_clip(img, frames: int, fps: int = LIVE_FPS) -> bytes:
    """The same RGB image repeated `frames` times as an H.264 mp4 (crf 18)."""
    import av
    out = io.BytesIO()
    with av.open(out, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = img.width, img.height, "yuv420p"
        st.options = {"crf": "18"}
        frame = av.VideoFrame.from_image(img.convert("RGB"))
        for _ in range(frames):
            for pkt in st.encode(frame):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return out.getvalue()


def _jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def to_static_sample(key: str, files: dict[str, bytes], meta: dict, frames: int,
                     size: tuple[int, int] = LIVE_SIZE) -> Sample | None:
    """One r2i triple -> an iv2v static-clip Sample, or None if it is not a usable train triple."""
    from PIL import Image
    g, p, t = _pick(files, "0"), _pick(files, "1"), _pick(files, "2")
    if not (g and p and t):
        return None  # i2i without a garment: the live session always has one
    m = meta.get("metadata", {})
    if m.get("split", "train") != "train":
        return None
    w, h = size
    target = Image.open(io.BytesIO(t)).convert("RGB")
    person = Image.open(io.BytesIO(p)).convert("RGB")
    if person.size != target.size:
        person = person.resize(target.size, Image.LANCZOS)
    extra = {k: m[k] for k in ("category", "config", "pseudo_source", "identity_pair", "input") if k in m}
    extra.update({"static_clip": True, "static_frames": frames, "from_key": key})
    return Sample(
        key=f"{key}_s{frames}", task="iv2v",
        files={"0.jpg": _jpeg(Image.open(io.BytesIO(g))), "0.mp4": static_clip(cover(person, w, h), frames),
               "1.mp4": static_clip(cover(target, w, h), frames)},
        caption=meta.get("caption") or "", captions={"caption_detailed": meta.get("caption_detailed")},
        source=m.get("source", ""), split="train", garment_id=m.get("garment_id"),
        person_id=m.get("person_id"), consent_id=m.get("consent_id"), extra=extra)


def video_sample(key: str, files: dict[str, bytes], meta: dict) -> Sample | None:
    """--vivid-videos: an existing iv2v clip pair, copied unchanged."""
    if not {"0.jpg", "0.mp4", "1.mp4"} <= set(files):
        return None
    m = meta.get("metadata", {})
    if m.get("split", "train") != "train":
        return None
    extra = {k: m[k] for k in ("category", "config", "pseudo_source", "identity_pair") if k in m}
    return Sample(key=key, task="iv2v", files={k: files[k] for k in ("0.jpg", "0.mp4", "1.mp4")},
                  caption=meta.get("caption") or "", captions={"caption_detailed": meta.get("caption_detailed")},
                  source=m.get("source", ""), split="train", garment_id=m.get("garment_id"),
                  person_id=m.get("person_id"), consent_id=m.get("consent_id"), extra={**extra, "flicker_risk": m.get("input") != "live_source"})


def detail_score(garment_bytes: bytes) -> float:
    """Edge energy of the garment photo's non-white pixels (eval.metrics): high = prints, trims,
    ribs, ruffles; low = plain fabric."""
    from PIL import Image
    a = metrics._arr(Image.open(io.BytesIO(garment_bytes)))
    m = a.min(axis=2) < 0.92
    return metrics._edge_energy(a, m if m.sum() > 100 else None)


def oversample_keys(scores: dict[str, float], factor: int) -> dict[str, int]:
    """key -> how many times to use it: the top third by score `factor` times, the rest once."""
    if factor <= 1 or not scores:
        return {k: 1 for k in scores}
    ranked = sorted(scores, key=lambda k: (-scores[k], k))
    top = set(ranked[: max(1, len(ranked) // 3)])
    return {k: (factor if k in top else 1) for k in scores}


def staging_index(dirs: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in dirs or []:
        root = Path(d)
        if root.is_dir():
            for p in root.iterdir():
                if p.is_dir():
                    out[p.name] = p
    return out


def staged_mask(index: dict[str, Path], key: str):
    """The staged mask.png of a shard key (identity pairs '<key>_id' share their pair's mask)."""
    from PIL import Image
    for k in (key, key[:-3] if key.endswith("_id") else None):
        if k and k in index and (index[k] / "mask.png").exists():
            return Image.open(index[k] / "mask.png")
    return None


def write_mask(mask_img, target_size: tuple[int, int], path: Path, size: tuple[int, int] = LIVE_SIZE) -> None:
    """Same cover crop as the clips: the mask is first brought to the target's size."""
    from PIL import Image
    m = mask_img.convert("L")
    if m.size != target_size:
        m = m.resize(target_size, Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cover(m, *size).save(path)


def allowed(source: str, mode: str) -> bool:
    try:
        src = sources.get(source)
    except Exception:
        return False
    return mode == "research" or src.commercial_ok


def build(manifest_dir: str, out: str, frames: int = 25, max_items: int = 3000, mode: str = "research",
          vivid_videos: bool = False, seed: int = 0, per_shard: int = 250,
          staging_dirs: list[str] | None = None, detail_oversample: int = 2,
          prefer_sources: tuple[str, ...] = ("vivid",), clip_repeat: int = 1) -> dict:
    if frames < 9 or (frames - 1) % 8:
        raise SystemExit("--frames must be 8n+1 and at least 9 (the VAE's time factor is 8)")
    if mode not in ("research", "commercial"):
        raise SystemExit("--mode must be research or commercial")
    mdir, out_dir = Path(manifest_dir), Path(out)
    stills = [(k, f, j) for k, f, j in iter_manifest(mdir, "i2i")]
    random.Random(seed).shuffle(stills)
    # Real-frame sources first (owner 2026-11-05: P8L showed synthetic Glance-only data pulls real
    # people toward the synthetic look), then the rest up to max_items; order within each stays random.
    stills.sort(key=lambda t: 0 if t[2].get("metadata", {}).get("source", "") in prefer_sources else 1)
    by_source: dict[str, list[Sample]] = defaultdict(list)
    counts = {"stills_seen": len(stills), "static_clips": 0, "skipped": 0, "dropped_by_mode": 0, "vivid_videos": 0,
              "masks": 0, "oversampled_extra": 0, "by_source": {}}
    index = staging_index(staging_dirs)
    scores: dict[str, float] = {}
    built: list[Sample] = []
    for key, files, meta in stills:
        if counts["static_clips"] >= max_items:
            break
        src = meta.get("metadata", {}).get("source", "")
        if not allowed(src, mode):
            counts["dropped_by_mode"] += 1
            continue
        s = to_static_sample(key, files, meta, frames)
        if s is None or not s.caption.strip():
            counts["skipped"] += 1
            continue
        built.append(s)
        counts["by_source"][src] = counts["by_source"].get(src, 0) + 1
        scores[s.key] = detail_score(_pick(files, "0"))
        mask = staged_mask(index, key)
        if mask is not None:
            from PIL import Image
            write_mask(mask, Image.open(io.BytesIO(_pick(files, "2"))).size, out_dir / "masks" / f"{s.key}.png")
            counts["masks"] += 1
        counts["static_clips"] += 1
    times = oversample_keys(scores, detail_oversample)
    for s in built:
        by_source[s.source].append(s)
        for n in range(2, times[s.key] + 1):
            dup = Sample(key=f"{s.key}_d{n}", task=s.task, files=s.files, caption=s.caption, source=s.source,
                         split=s.split, garment_id=s.garment_id, person_id=s.person_id, consent_id=s.consent_id,
                         extra={**s.extra, "detail_oversampled": True}, captions=dict(s.captions))
            by_source[s.source].append(dup)
            m = out_dir / "masks" / f"{s.key}.png"
            if m.exists():
                (out_dir / "masks" / f"{dup.key}.png").write_bytes(m.read_bytes())
            counts["oversampled_extra"] += 1
    if vivid_videos:
        for key, files, meta in iter_manifest(mdir, "v2v"):
            src = meta.get("metadata", {}).get("source", "")
            if not allowed(src, mode):
                counts["dropped_by_mode"] += 1
                continue
            s = video_sample(key, files, meta)
            if s is not None and s.caption.strip():
                by_source[s.source].append(s)
                counts["vivid_videos"] += 1
                # Real-motion clips (owner 2026-11-05): repeat each so motion is a fair share of the
                # steps; the loader takes a new random 25-frame window of the clip each time.
                for r in range(2, clip_repeat + 1):
                    by_source[s.source].append(Sample(key=f"{s.key}_r{r}", task=s.task, files=s.files, caption=s.caption,
                                                      source=s.source, split=s.split, garment_id=s.garment_id,
                                                      person_id=s.person_id, consent_id=s.consent_id,
                                                      extra={**s.extra, "clip_repeat": r}, captions=dict(s.captions)))
                    counts["vivid_videos"] += 1
    catalog = []
    for source, items in sorted(by_source.items()):
        catalog += write_shards(items, out_dir / "shards", f"joyai0811-{source}", per_shard)
    selection = manifests.build(catalog, mode, "train")
    manifests.write(selection, out_dir / "manifest")
    vals = sorted(scores.values())
    detail = ({"n": len(vals), "min": round(vals[0], 5), "median": round(vals[len(vals) // 2], 5),
               "max": round(vals[-1], 5), "oversample": detail_oversample,
               "top_third_threshold": round(vals[-max(1, len(vals) // 3)], 5)} if vals else {})
    summary = {**counts, "detail_scores": detail, "frames": frames, "size": list(LIVE_SIZE), "mode": mode,
               "shards": len(catalog), "mask_dir": str(out_dir / "masks"),
               "manifest_dir": str(out_dir / "manifest"),
               **{k: selection[k] for k in ("used_sources", "commercial_use_cleared", "trained_with_vivid")}}
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=25, help="frames per static clip, 8n+1 (25 = 4 latent chunks)")
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--mode", default="research", choices=["research", "commercial"])
    ap.add_argument("--vivid-videos", action="store_true",
                    help="also train on the ViViD clip pairs (they flicker; owner opt-in only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-shard", type=int, default=250)
    ap.add_argument("--staging-dirs", nargs="*", default=[], help="staged pair folders (for mask.png)")
    ap.add_argument("--detail-oversample", type=int, default=2, help="repeat the most detailed third N times (1 = off)")
    ap.add_argument("--clip-repeat", type=int, default=1, help="with --vivid-videos: use each clip N times")
    ap.add_argument("--prefer-source", nargs="*", default=["vivid"],
                    help="sources taken first, before the rest fills --max (real frames first; empty = random)")
    a = ap.parse_args(argv)
    summary = build(a.manifest_dir, a.out, a.frames, a.max, a.mode, a.vivid_videos, a.seed, a.per_shard,
                    a.staging_dirs, a.detail_oversample, tuple(a.prefer_source), a.clip_repeat)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
