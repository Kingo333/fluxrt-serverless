"""Short runs of consecutive frames where the clothes move (fxreal6, owner 2026-11-15).

Live FluxRT draws about 6 pictures a second, so two pictures in a row are about 0.17 s apart. A
"window" is K frames that far apart (K=4: about half a second of a step, a turn or a swinging hem).
The trainer keeps a window in one batch with the same noise, and its sequence term lets the garment,
face and skin change between the frames only as much as the real frames do.

Pure functions here (the masks come from data_tools.vivid_motion's segmentation pass):

  stride_for(fps)                      frames to skip between two pictures of a window
  pick_windows(masks, k, stride, n)    the n windows with the most clothes movement, not overlapping
"""

from __future__ import annotations

from .vivid_motion import change

LIVE_GAP_S = 0.17  # 1 / 6 base frames per second


def stride_for(fps: float, gap_s: float = LIVE_GAP_S) -> int:
    """Source frames between two pictures of a window (at least 1)."""
    if not fps or fps <= 0:
        return 5  # 30 fps footage
    return max(1, round(fps * gap_s))


def window_frames(start: int, k: int, stride: int) -> list[int]:
    return [start + i * stride for i in range(k)]


def window_motion(masks: dict, frames: list[int]) -> float | None:
    """Clothes movement over a window: the sum of the change between neighbouring frames, or None when
    a frame of the window was not segmented (too small a person, or past the end of the clip)."""
    if any(t not in masks for t in frames):
        return None
    return sum(change(masks[a], masks[b]) for a, b in zip(frames, frames[1:]))


def pick_windows(masks: dict, k: int = 4, stride: int = 5, n: int = 2, min_motion: float = 0.0) -> list[dict]:
    """Up to n windows of k frames, `stride` apart, ranked by clothes movement; windows never share a
    frame span, so two windows of one clip show two different moments. Earliest first on a tie."""
    if k < 2 or stride < 1 or n < 1:
        return []
    scored = []
    for start in sorted(masks):
        fr = window_frames(start, k, stride)
        m = window_motion(masks, fr)
        if m is not None and m >= min_motion:
            scored.append((m, start, fr))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked: list[dict] = []
    for m, start, fr in scored:
        if all(fr[-1] < p["frames"][0] or fr[0] > p["frames"][-1] for p in picked):
            picked.append({"frames": fr, "motion": round(m, 4)})
            if len(picked) == n:
                break
    return sorted(picked, key=lambda p: p["frames"][0])


# ---------------------------------------------------------------- the tool (MediaPipe venv)
def pair_folders(window: list[int]) -> list[tuple[str, tuple[int, int]]]:
    """A 4-frame window as two frame pairs, the layout build_real's pair code already reads:
    (suffix, (frame a, frame b)). The trainer keeps each pair in one batch with the same noise."""
    return [("a", (window[0], window[1])), ("b", (window[2], window[3]))]


def main(argv=None) -> dict:
    """python -m data_tools.sequences STAGING/vivid OUT [--windows 2] [--size 432x768] [--model SEG.tflite]

    Every ViViD TRAIN clip of every garment type: scores the clothes movement on every 2nd frame, takes
    the --windows moments (4 frames, one live frame gap apart) where the clothes move most, and writes
    each as two pair folders OUT/<clip>_w<k>a and OUT/<clip>_w<k>b with a_target.png, a_mask.png,
    a_skin.png, b_*.png, ref.jpg and meta.json (category, clip, frames, motion, garment_description)."""
    import argparse
    import json
    import shutil
    from pathlib import Path
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vivid")
    ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--n", type=int, default=0, help="clips to use (0 = every train clip)")
    ap.add_argument("--size", default="432x768")
    ap.add_argument("--min-motion", type=float, default=0.02, help="skip windows where the clothes hardly move")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    a = ap.parse_args(argv)
    import av
    import numpy as np
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    from data_tools.vivid_motion import BACKGROUND, CLOTHES, MIN_CLOTHES_PX, MIN_PERSON_FRAC, _cover
    BODY_SKIN = 2
    seg0 = Segmenter(Path(a.model))
    seg = lambda im: np.squeeze(seg0(im))  # noqa: E731
    w, h = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for d in sorted(p for p in Path(a.vivid).iterdir() if p.is_dir()):
        mp = d / "meta.json"
        if mp.exists() and (d / "target.mp4").exists() and (d / "ref.jpg").exists():
            meta = json.loads(mp.read_text())
            if meta.get("split") == "train":
                clips.append((d, meta))
    stats = {"clips": len(clips), "clips_used": 0, "windows": 0, "pair_folders": 0, "kept_existing": 0, "skipped": 0,
             "by_type": {}, "with_description": 0}
    for d, meta in clips:
        if a.n and stats["clips_used"] >= a.n:
            break
        if (out / f"{d.name}.done").exists():
            stats["kept_existing"] += 1
            stats["clips_used"] += 1
            continue
        with av.open(str(d / "target.mp4")) as c:
            fps = float(c.streams.video[0].average_rate or 25)
            frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
        stride = stride_for(fps)
        every = 2 if stride % 2 == 0 else 1
        masks = {}
        for t in range(0, len(frames), every):
            cat = seg(_cover(frames[t], w // 2, h // 2))
            if float((cat != BACKGROUND).mean()) >= MIN_PERSON_FRAC and int((cat == CLOTHES).sum()) >= MIN_CLOTHES_PX // 4:
                masks[t] = cat == CLOTHES
        wins = pick_windows(masks, k=4, stride=stride, n=a.windows, min_motion=a.min_motion)
        if not wins:
            stats["skipped"] += 1
            (out / f"{d.name}.done").write_text("no window")
            continue
        desc = str(meta.get("garment_description") or "").strip()
        for k, win in enumerate(wins, 1):
            for suffix, pair in pair_folders(win["frames"]):
                dst = out / f"{d.name}_w{k}{suffix}"
                dst.mkdir(parents=True, exist_ok=True)
                for tag, t in zip("ab", pair):
                    im = _cover(frames[t], w, h)
                    cat = seg(im)
                    im.save(dst / f"{tag}_target.png")
                    Image.fromarray(((cat == CLOTHES) * 255).astype(np.uint8), "L").save(dst / f"{tag}_mask.png")
                    Image.fromarray(((cat == BODY_SKIN) * 255).astype(np.uint8), "L").save(dst / f"{tag}_skin.png")
                shutil.copy(d / "ref.jpg", dst / "ref.jpg")
                (dst / "meta.json").write_text(json.dumps({
                    "category": meta.get("category", "upper"), "clip": d.name, "window": k, "frames": list(pair),
                    "motion": win["motion"], "stride": stride, "split": "train", "source": "vivid",
                    **({"garment_description": desc} if desc else {})}))
                stats["pair_folders"] += 1
            stats["windows"] += 1
        stats["clips_used"] += 1
        stats["with_description"] += int(bool(desc))
        cat_name = meta.get("category", "upper")
        stats["by_type"][cat_name] = stats["by_type"].get(cat_name, 0) + 1
        (out / f"{d.name}.done").write_text("ok")
        if stats["clips_used"] % 100 == 0:
            print(json.dumps({"sequence_progress": stats["clips_used"], "windows": stats["windows"]}), flush=True)
    print("SEQUENCES " + json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
