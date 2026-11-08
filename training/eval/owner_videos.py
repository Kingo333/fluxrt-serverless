"""Five blind side-by-side videos for the owner (2026-11-08): one per test case.

  python -m eval.owner_videos RESULTS/cases --gate RESULTS/gate_summary.json --out EVAL/owner_videos \
      [--base base] [--candidates l2000 l1500 l2000s07 l1000] [--height 640]

Each video = every clip of one case (standing, turning, arms, lighting, feet) one after another,
each clip played twice, with four panels:  garment | camera | A | B.
A and B are today's FluxRT (base) and the best LoRA save, in an order shuffled per video; the key is
written to OUT/key.json (keep it from the viewer until they have picked). "Best save" = the first of
--candidates that passed the face/skin/background gate (gate_summary.json); if none passed, the first
candidate, and key.json says so. H.264, yuv420p, 24 fps: plays on any phone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CASES = ("standing", "turning", "arms", "lighting", "feet")


def pick_candidate(candidates: list[str], gate: dict) -> tuple[str, bool]:
    for c in candidates:
        if gate.get(c, {}).get("pass"):
            return c, True
    return candidates[0], False


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _frames(path: Path):
    import av
    with av.open(str(path)) as c:
        return [f.to_image().convert("RGB") for f in c.decode(video=0)]


def _panel(img, w: int, h: int, label: str):
    from PIL import Image, ImageDraw
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, w, 40), fill=(0, 0, 0))
    d.text((12, 6), label, fill=(255, 255, 255), font=_font(26))
    return canvas


def build_case(clips: list[Path], base: str, cand: str, left_is_base: bool, out: Path, height: int, repeats: int) -> int:
    import av
    from PIL import Image
    pw = height * 9 // 16 // 2 * 2
    W, H = pw * 4, height
    frames_written = 0
    with av.open(str(out), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "20", "preset": "medium"}
        for d in clips:
            cam = _frames(d / "input.mp4")
            a_p, b_p = (d / f"out_{base}.mp4", d / f"out_{cand}.mp4")
            if not left_is_base:
                a_p, b_p = b_p, a_p
            A, B = _frames(a_p), _frames(b_p)
            ref = next((p for p in d.iterdir() if p.stem == "ref"), None)
            g = _panel(Image.open(ref).convert("RGB"), pw, H, "garment") if ref else Image.new("RGB", (pw, H))
            n = min(len(cam), len(A), len(B))
            for _ in range(repeats):
                for k in range(n):
                    row = Image.new("RGB", (W, H))
                    row.paste(g, (0, 0))
                    row.paste(_panel(cam[k], pw, H, "camera"), (pw, 0))
                    row.paste(_panel(A[k], pw, H, "A"), (2 * pw, 0))
                    row.paste(_panel(B[k], pw, H, "B"), (3 * pw, 0))
                    for pkt in s.encode(av.VideoFrame.from_image(row)):
                        c.mux(pkt)
                    frames_written += 1
        for pkt in s.encode():
            c.mux(pkt)
    return frames_written


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases_dir")
    ap.add_argument("--gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--candidates", nargs="+", default=["l2000", "l1500", "l2000s07", "l1000"])
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    gate = json.loads(Path(a.gate).read_text()) if a.gate and Path(a.gate).exists() else {}
    cand, passed = pick_candidate(a.candidates, gate)
    root, out = Path(a.cases_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    key = {"base": a.base, "lora": cand, "lora_passed_gate": passed, "videos": {}}
    for i, case in enumerate(CASES, 1):
        clips = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith(f"{case}_")
                 and (d / "input.mp4").exists() and (d / f"out_{a.base}.mp4").exists() and (d / f"out_{cand}.mp4").exists()]
        name = f"{i}_{case}.mp4"
        if not clips:
            key["videos"][name] = {"missing": f"no {case} clip with both {a.base} and {cand} outputs"}
            continue
        left_is_base = rng.random() < 0.5
        n = build_case(clips, a.base, cand, left_is_base, out / name, a.height, a.repeats)
        key["videos"][name] = {"A": a.base if left_is_base else cand, "B": cand if left_is_base else a.base,
                               "clips": [d.name for d in clips], "frames": n}
    (out / "key.json").write_text(json.dumps(key, indent=2))
    (out / "README.txt").write_text(
        "Five test cases: 1 standing, 2 turning, 3 arms up, 4 different light, 5 feet/shoes.\n"
        "Panels: garment | camera | A | B. One of A/B is today's FluxRT, the other the new LoRA (order differs per video).\n"
        "For each video pick: which is better at the GARMENT, and is either worse at face, skin tone or background?\n"
        "key.json says which is which: open it only after picking.\n")
    print(json.dumps({k: v for k, v in key.items() if k != "videos"} | {"made": [n for n, v in key["videos"].items() if "missing" not in v]}))
    return key


if __name__ == "__main__":
    main()
