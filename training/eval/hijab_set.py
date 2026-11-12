"""The hijab test set (owner, 2026-11-11): a shopper who wears a hijab tries on a dress or abaya; her own hijab
must stay exactly as it is.

  python -m eval.hijab_set ROOT --out ROOT/eval/fxa/hijab.json [--own ROOT/eval/own_clips]
      [--clips-dir ROOT/eval/hijab_clips] [--refs-from SET.json ...] [--n 3] [--model SEG.tflite]

Clips, in this order, until --n are kept:
- the owner's own clips: any .mp4/.mov/.m4v in --own (e.g. a phone video of a consenting hijab-wearing person
  standing in front of the camera, like a shopper);
- free stock clips (Pexels licence: free to use, no attribution needed; source and author are kept in the set
  file), downloaded once into --clips-dir.

Each clip is checked, not trusted: MediaPipe's multiclass selfie segmenter must find a face with a head
covering (clothes / other pixels around and above the face outweighing hair; eval.preserve_check.head_zone)
and a person filling at least 15% of the 9:16 frame, in at least 70% of the sampled frames of a 49-frame
window. The best window is written as <clips-dir>/<key>_test.mp4 (49 frames, 576 px wide).

Garment photos: dresses from the given eval sets (the owner's set2 first: two long black maxi dresses), cleaned
like the live dress cleaner (FLUXRT_REF_CLEAN_DRESS: clothes only, nothing above the chin) when a person is on
the photo, so the test sees what live FluxRT sees. Falls back to held-out ViViD dress photos.

Writes the set file (the same item format as the other eval sets: key, case, video, ref, category, name,
description, plus source and author) and prints a summary; prints NO_HIJAB_CLIP when none passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACE, BODY_SKIN, HAIR, CLOTHES, BACKGROUND = 3, 2, 1, 4, 0
WINDOW = 49           # frames, like the other eval clips
SAMPLE_EVERY = 6      # frames between two checked frames
MIN_OK = 0.7          # share of checked frames in the window that must pass
MIN_PERSON = 0.15     # person share of the 9:16 frame
MAX_SECONDS = 40      # only the start of a long clip is searched
MAX_BYTES = 400 * 1024 * 1024

# Free stock clips (Pexels licence). Direct file links first; pexels.com/download links are a fallback.
STOCK = [
    ("pexels_8734578", "https://videos.pexels.com/video-files/8734578/8734578-uhd_1440_2732_25fps.mp4",
     "https://www.pexels.com/video/8734578/", "Ron Lach (Pexels)", "full body, standing with a skateboard"),
    ("pexels_8748836", "https://videos.pexels.com/video-files/8748836/8748836-uhd_1440_2732_24fps.mp4",
     "https://www.pexels.com/video/8748836/", "Anastasia Shuraeva (Pexels)", "standing against a plain backdrop"),
    ("pexels_7896885", "https://videos.pexels.com/video-files/7896885/7896885-hd_1080_1920_25fps.mp4",
     "https://www.pexels.com/video/7896885/", "Darina Belonogova (Pexels)", "adjusting her look"),
    ("pexels_7728223", "https://www.pexels.com/download/video/7728223/",
     "https://www.pexels.com/video/7728223/", "Monstera Production (Pexels)", "pink hijab, studio"),
    ("pexels_6245808", "https://www.pexels.com/download/video/6245808/",
     "https://www.pexels.com/video/6245808/", "Monstera Production (Pexels)", "standing indoors, shelves behind"),
]
VIDEO_EXT = (".mp4", ".mov", ".m4v")


def cover_9x16(img, w: int = 288, h: int = 512):
    """Centre crop to 9:16 and resize, like eval.generate_fluxrt does with every camera frame."""
    from PIL import Image
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.BILINEAR)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def frame_ok(cat) -> bool:
    """A person filling enough of the frame, with a face and a head covering."""
    from eval.preserve_check import head_zone
    if float((cat != BACKGROUND).mean()) < MIN_PERSON:
        return False
    _, cover = head_zone(cat)
    return cover is not None


def best_window(flags: list[bool], per_window: int) -> tuple[int, float]:
    """(start index into flags, share of passing flags) of the best run of per_window flags; earliest on ties."""
    if not flags:
        return 0, 0.0
    per_window = max(1, min(per_window, len(flags)))
    best = (0, -1.0)
    for i in range(0, len(flags) - per_window + 1):
        share = sum(flags[i:i + per_window]) / per_window
        if share > best[1]:
            best = (i, share)
    return best


def clean_dress_photo(rgb, cat):
    """The live dress cleaner's rule (livecam.segment.clean_reference_dress): when a person is on the photo,
    keep the clothes only and nothing above the chin, on white. None when there is no person to clean."""
    import numpy as np
    if cat.shape != rgb.shape[:2] or float(np.isin(cat, (FACE, BODY_SKIN, HAIR)).mean()) < 0.005:
        return None
    garment = cat == CLOTHES
    rows = np.nonzero((cat == FACE).any(axis=1))[0]
    if rows.size:
        garment[: int(rows.max()) + 1] = False
    if garment.mean() < 0.02:
        return None
    out = np.full_like(rgb, 255)
    out[garment] = rgb[garment]
    return out


def download(url: str, dest: Path) -> str | None:
    """None when the file is there, else the reason it is not."""
    import urllib.request
    if dest.exists() and dest.stat().st_size > 100_000:
        return None
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                                               "Referer": "https://www.pexels.com/"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            n = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > MAX_BYTES:
                    return "too big"
                f.write(chunk)
        if n < 100_000:
            return f"only {n} bytes"
        tmp.replace(dest)
        return None
    except Exception as e:  # noqa: BLE001 - one bad link must not stop the set
        return f"{type(e).__name__}: {e}"
    finally:
        tmp.unlink(missing_ok=True)


def scan(path: Path, seg) -> dict:
    """Check every SAMPLE_EVERY-th frame of the first MAX_SECONDS; pick the best 49-frame window."""
    import av
    import numpy as np
    flags, fps = [], 25.0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate or 25)
        limit = int(fps * MAX_SECONDS)
        for k, f in enumerate(c.decode(st)):
            if k >= limit:
                break
            if k % SAMPLE_EVERY == 0:
                flags.append(frame_ok(np.squeeze(seg(cover_9x16(f.to_image().convert("RGB"))))))
    i, share = best_window(flags, -(-WINDOW // SAMPLE_EVERY))
    return {"start": i * SAMPLE_EVERY, "share": round(share, 3), "checked": len(flags), "fps": fps}


def write_window(src: Path, dest: Path, start: int, n: int = WINDOW, width: int = 576) -> int:
    import av
    written = 0
    with av.open(str(src)) as c, av.open(str(dest), "w") as o:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        out = None
        for k, f in enumerate(c.decode(st)):
            if k < start:
                continue
            if written >= n:
                break
            img = f.to_image().convert("RGB")
            h = round(img.height * width / img.width) // 2 * 2
            img = img.resize((width, h))
            if out is None:
                out = o.add_stream("libx264", rate=int(round(float(st.average_rate or 25))))
                out.width, out.height, out.pix_fmt = width, h, "yuv420p"
                out.options = {"crf": "16", "preset": "medium"}
            for pkt in out.encode(av.VideoFrame.from_image(img)):
                o.mux(pkt)
            written += 1
        if out is not None:
            for pkt in out.encode():
                o.mux(pkt)
    return written


def dress_refs(root: Path, sets: list[Path], seg, out_dir: Path, k: int = 2) -> list[dict]:
    """Up to k dress photos from the eval sets (in order), cleaned like live; else held-out ViViD dresses."""
    import numpy as np
    from PIL import Image
    cands, seen = [], set()
    for s in sets:
        if s and Path(s).exists():
            for it in json.loads(Path(s).read_text()):
                if str(it.get("category", "")).lower() in ("dresses", "dress") and it.get("ref") and it["ref"] not in seen:
                    seen.add(it["ref"])
                    cands.append({"ref": it["ref"], "name": it.get("name"), "description": it.get("description"),
                                  "from": Path(s).stem})
    if len(cands) < k:
        staging = root / "data/staging/vivid"
        for mp in sorted(staging.glob("*/meta.json")) if staging.is_dir() else []:
            meta = json.loads(mp.read_text())
            ref = mp.parent / "ref.jpg"
            if meta.get("category") == "dress" and meta.get("split") != "train" and ref.exists() and str(ref) not in seen:
                cands.append({"ref": str(ref), "name": None, "description": meta.get("garment_description"),
                              "from": "vivid_heldout"})
                seen.add(str(ref))
            if len(cands) >= k:
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for i, c in enumerate(cands[:k], 1):
        img = Image.open(c["ref"]).convert("RGB")
        small = img.copy()
        small.thumbnail((512, 512))
        cat = np.squeeze(seg(small)).astype(np.uint8)                       # segment small, clean at full size
        cat = np.asarray(Image.fromarray(cat).resize(img.size, Image.NEAREST))
        cleaned = clean_dress_photo(np.asarray(img), cat)
        dest = out_dir / f"dress{i}.png"
        (Image.fromarray(cleaned) if cleaned is not None else img).save(dest)
        refs.append({**c, "ref": str(dest), "source_ref": c["ref"], "cleaned": cleaned is not None})
    return refs


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--own", help="folder with the owner's own clips (optional)")
    ap.add_argument("--clips-dir")
    ap.add_argument("--refs-from", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-stock", action="store_true", help="only the owner's own clips")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    a = ap.parse_args(argv)
    from data_tools.clothes_mask import Segmenter
    root = Path(a.root)
    clips_dir = Path(a.clips_dir or root / "eval/hijab_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(Path(a.model))
    refs = dress_refs(root, [Path(p) for p in a.refs_from], seg, clips_dir / "refs")
    summary = {"refs": [{k: r[k] for k in ("ref", "from", "cleaned")} for r in refs], "clips": [], "rejected": {}}
    if not refs:
        print("NO_HIJAB_CLIP: no dress photo to try on (no dresses in the eval sets, no held-out ViViD dress)")
        Path(a.out).write_text("[]")
        return summary
    cands = []
    own = Path(a.own) if a.own else None
    if own and own.is_dir():
        for p in sorted(own.iterdir()):
            if p.suffix.lower() in VIDEO_EXT:
                cands.append((f"own_{p.stem}", p, "owner clip", "owner", ""))
    if not a.no_stock:
        for key, url, page, author, note in STOCK:
            dest = clips_dir / f"{key}.mp4"
            why = download(url, dest)
            if why:
                summary["rejected"][key] = f"download failed: {why}"
                continue
            cands.append((key, dest, page, author, note))
    items = []
    for key, path, source, author, note in cands:
        if len(items) >= a.n:
            break
        try:
            r = scan(path, seg)
        except Exception as e:  # noqa: BLE001 - an unreadable clip is skipped
            summary["rejected"][key] = f"unreadable: {type(e).__name__}: {e}"
            continue
        if r["share"] < MIN_OK:
            summary["rejected"][key] = f"head covering + person in only {r['share']:.0%} of the best window"
            continue
        test = clips_dir / f"{key}_test.mp4"
        if write_window(path, test, r["start"]) < WINDOW:
            summary["rejected"][key] = "clip too short after the best window"
            continue
        ref = refs[len(items) % len(refs)]
        items.append({"key": f"hijab{len(items) + 1}_{key}", "case": "hijab", "video": str(test), "ref": ref["ref"],
                      "category": "dresses", "name": ref.get("name"), "description": ref.get("description"),
                      "source": source, "author": author, "note": note, "window_start": r["start"],
                      "window_share": r["share"]})
        summary["clips"].append({"key": items[-1]["key"], "source": source, "share": r["share"]})
    Path(a.out).write_text(json.dumps(items, indent=2))
    print(json.dumps(summary, indent=1))
    if items:
        print(f"HIJAB_CLIPS {len(items)}: " + ", ".join(f"{it['key']} ({it['source']})" for it in items))
    else:
        print("NO_HIJAB_CLIP: no clip passed the check. Put a short phone clip of a hijab-wearing person "
              f"(with her consent) in {own or root / 'eval/own_clips'} and run again.")
    return summary


if __name__ == "__main__":
    main()
