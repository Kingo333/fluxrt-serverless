"""The owner's 5-item re-test (2026-11-08): 5 clips, prompts that match each garment photo, one blind video.

  python five_test.py select ROOT [--desc N=TEXT ...] [--category N=tops|bottoms|dresses ...]
  python five_test.py video  ROOT [--lora l2000] [--seed 7]

select: takes the 8 case items (eval/fx2/cases/clip_list.json, after fix_case_text.py), keeps 5
  (one per case first, then the most different garment category), applies the owner's corrections
  (--desc 2="plain white wide-leg trousers"), writes eval/fx2/cases5/clip_list.json and PRINTS the
  exact prompt of each item. Nothing is rendered.
video: one H.264 video, eval/five_test/five_test.mp4: per item a segment garment | camera | A | B
  (each segment played twice), A/B order random per segment. The key goes to a separate folder,
  eval/five_test_KEY/key.json, so the video folder can be shared without it.
Run from the training code checkout (needs data_tools.fluxrt_prompt and eval.owner_videos).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

ORDER = ("standing", "turning", "arms", "lighting", "feet")


def pick_five(items: list[dict], n: int = 5) -> list[dict]:
    """One item per case first (preferring a new category, then a new garment), then fill to n the same way."""
    chosen: list[dict] = []

    def score(it):  # lower is better
        return (it["category"] in {c["category"] for c in chosen},
                it.get("garment_key") in {c.get("garment_key") for c in chosen})

    for case in ORDER:
        cands = [it for it in items if it.get("case") == case and it not in chosen]
        if cands and len(chosen) < n:
            chosen.append(min(cands, key=score))
    while len(chosen) < n:
        rest = [it for it in items if it not in chosen]
        if not rest:
            break
        chosen.append(min(rest, key=score))
    return chosen


def _overrides(pairs: list[str] | None) -> dict[int, str]:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[int(k)] = v.strip()
    return out


def select(root: Path, descs: list[str] | None, cats: list[str] | None) -> list[dict]:
    from data_tools.fluxrt_prompt import compact_prompt
    e = root / "eval/fx2"
    items = json.loads((e / "cases/clip_list.json").read_text())
    garments = {g["key"]: g for g in
                json.loads((root / "azyah-code/training/eval/test_garments.json").read_text())["items"]}
    five = pick_five(items)
    d, c = _overrides(descs), _overrides(cats)
    for i, it in enumerate(five, 1):
        if i in d:
            it["description"] = d[i] or None
        if i in c:
            it["category"] = c[i]
        it["key"] = f"{i}_{it['key']}"
        it["prompt"] = compact_prompt(it.get("category"), it.get("name"), it.get("description"))
    out = e / "cases5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "clip_list.json").write_text(json.dumps(five, indent=2))
    for i, it in enumerate(five, 1):
        g = garments.get(it.get("garment_key"), {})
        print(f"\n[{i}] case={it.get('case')} clip={Path(it['video']).parent.name} category={it['category']}"
              f" photo={it['ref']} (from {g.get('photo_from') or 'catalogue'})\n    PROMPT: {it['prompt']}")
    print(f"\nwrote {out / 'clip_list.json'} ({len(five)} items). Fix a line with --desc N=\"...\" or --category N=...")
    return five


def video(root: Path, lora: str, seed: int, height: int = 640, repeats: int = 2, results: str | None = None,
          name: str = "five_test") -> dict:
    import av
    from PIL import Image
    from eval.owner_videos import _frames, _panel
    res = Path(results) if results else root / "eval/fx2/results/cases5"
    items = json.loads((root / "eval/fx2/cases5/clip_list.json").read_text())
    out, key_dir = root / "eval" / name, root / "eval" / f"{name}_KEY"
    out.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pw = height * 9 // 16 // 2 * 2
    W, H = pw * 4, height
    key = {"lora": lora, "base": "base", "segments": []}
    with av.open(str(out / f"{name}.mp4"), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "20", "preset": "medium"}
        for i, it in enumerate(items, 1):
            d = res / it["key"]
            paths = {"base": d / "out_base.mp4", lora: d / f"out_{lora}.mp4"}
            if not all(p.exists() for p in paths.values()) or not (d / "input.mp4").exists():
                key["segments"].append({"n": i, "item": it["key"], "missing": "render not found"})
                continue
            a_model = "base" if rng.random() < 0.5 else lora
            b_model = lora if a_model == "base" else "base"
            cam, A, B = _frames(d / "input.mp4"), _frames(paths[a_model]), _frames(paths[b_model])
            g = _panel(Image.open(it["ref"]).convert("RGB"), pw, H, f"{i}/{len(items)} garment")
            for _ in range(repeats):
                for k in range(min(len(cam), len(A), len(B))):
                    row = Image.new("RGB", (W, H))
                    row.paste(g, (0, 0))
                    row.paste(_panel(cam[k], pw, H, "camera"), (pw, 0))
                    row.paste(_panel(A[k], pw, H, "A"), (2 * pw, 0))
                    row.paste(_panel(B[k], pw, H, "B"), (3 * pw, 0))
                    for pkt in s.encode(av.VideoFrame.from_image(row)):
                        c.mux(pkt)
            key["segments"].append({"n": i, "item": it["key"], "case": it.get("case"), "A": a_model, "B": b_model,
                                    "prompt": it.get("prompt")})
        for pkt in s.encode():
            c.mux(pkt)
    (key_dir / "key.json").write_text(json.dumps(key, indent=2))
    (out / "README.txt").write_text(
        f"{len(items)} items, one after another (the garment panel says 1/{len(items)}, 2/{len(items)}, ...).\n"
        "Panels: garment | camera | A | B. One of A/B is today's FluxRT, the other the new LoRA; the order changes "
        "per item.\nFor each item pick: which is better at the GARMENT, and is either worse at face, skin tone or "
        f"background?\nThe key is in a separate folder ({name}_KEY): open it only after picking.\n")
    made = [x["n"] for x in key["segments"] if "missing" not in x]
    print(json.dumps({"video": str(out / f"{name}.mp4"), "segments": made, "lora": lora,
                      "missing": [x["item"] for x in key["segments"] if "missing" in x]}))
    return key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("root", nargs="?", default="/workspace/azyah-vton")
    s.add_argument("--desc", action="append")
    s.add_argument("--category", action="append")
    v = sub.add_parser("video")
    v.add_argument("root", nargs="?", default="/workspace/azyah-vton")
    v.add_argument("--lora", default="l2000")
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--results", help="folder with <item>/out_base.mp4 and out_<lora>.mp4 (default eval/fx2/results/cases5)")
    v.add_argument("--name", default="five_test", help="video name; also the eval/<name> and eval/<name>_KEY folders")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return select(Path(a.root), a.desc, a.category)
    return video(Path(a.root), a.lora, a.seed, results=a.results, name=a.name)


if __name__ == "__main__":
    main()
