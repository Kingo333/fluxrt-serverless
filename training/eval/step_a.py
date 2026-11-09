"""Step A (owner, 2026-11-09): blends of fxreal3 and fxreal4, judged on 10 clips, every panel labelled.

  python -m eval.step_a clips  ROOT            the 10-clip list: the owner's 5 (eval/fx2/cases5) + 5 held-out
                                               (eval/fxa/descriptions.json {"7": "..."} adds owner descriptions)
                                               ViViD test clips, each trying on another held-out clip's garment
                                               of the same type -> eval/fxa/clips10.json (prints the prompts)
  python -m eval.step_a reuse  ROOT            copies the renders that already exist for the owner's 5 clips
                                               (original and fxreal4 from eval/fx4, fxreal3 from eval/fx3) into
                                               eval/fxa/results/clips10, so only new work is rendered
  python -m eval.step_a video  ROOT [--panels base=original fx3=fx3 ...] [--repeats 2]
                                               one labelled video of all clips: garment | camera | each model,
                                               names on every panel (not blind) -> eval/fxa/step_a_labelled.mp4
  python -m eval.step_a pick   ROOT --set NAME [--n 10]   NEW held-out clips (none used by any earlier list),
                                               garments from other held-out clips of the same type; writes
                                               eval/fxa/NAME.json and eval/fxa/NAME_descriptions.json (empty,
                                               for the owner to fill in) and prints each garment photo path
  python -m eval.step_a apply  ROOT --set NAME  requires a description for EVERY item (else exits 2),
                                               rebuilds the prompts like the live app, prints them
  python -m eval.step_a video  ROOT --set NAME --panels ... [--no-legend]   same video for another set
Run from the training code checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

APP_CATEGORY = {"upper": "tops", "lower": "bottoms", "dress": "dresses"}
NEW_CATEGORIES = ("upper", "upper", "dress", "lower", "upper")  # with the owner's 5: 4 tops, 3 dresses, 3 bottoms
DEFAULT_PANELS = ["base=original", "fx3=fx3", "fx4=fx4", "blend05=blend 0.5", "blend03=blend 0.3",
                  "blend05s08=blend 0.5 @0.8", "blend05s065=blend 0.5 @0.65"]


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def held_out_clips(vivid: Path) -> list[tuple[str, dict, Path]]:
    out = []
    for d in sorted(p for p in vivid.iterdir() if p.is_dir()):
        m = d / "meta.json"
        if m.exists() and (d / "target.mp4").exists() and (d / "ref.jpg").exists():
            meta = json.loads(m.read_text())
            if meta.get("split") == "test":
                out.append((d.name, meta, d))
    return out


def pick_new_clips(clips: list[tuple[str, dict, Path]], exclude: set, excluded_ref_shas: set,
                   categories=NEW_CATEGORIES, seed: int = 5) -> list[dict]:
    """5 held-out person clips (by category) and, for each, a garment from ANOTHER held-out clip of the same
    category: a real change to judge, with a ViViD garment photo and no catalogue text."""
    rng = random.Random(seed)
    pool = [c for c in clips if c[0] not in exclude and _sha(c[2] / "ref.jpg") not in excluded_ref_shas]
    rng.shuffle(pool)
    by_cat: dict[str, list] = {}
    for c in pool:
        by_cat.setdefault(c[1].get("category", "upper"), []).append(c)
    picked, used = [], set()
    for cat in categories:
        cands = [c for c in by_cat.get(cat, []) if c[0] not in used] or [c for c in pool if c[0] not in used]
        if not cands:
            break
        person = cands[0]
        pcat = person[1].get("category", "upper")
        donors = [c for c in by_cat.get(pcat, []) if c[0] != person[0] and c[0] not in used]
        if not donors:
            continue
        donor = donors[-1]
        used.update({person[0], donor[0]})
        picked.append({"person": person, "donor": donor})
    return picked


def build_clip_list(root: Path) -> list[dict]:
    from data_tools.fluxrt_prompt import compact_prompt
    owner = json.loads((root / "eval/fx2/cases5/clip_list.json").read_text())
    exclude = {Path(it["video"]).parent.name for it in owner}
    ref_shas = {_sha(Path(it["ref"])) for it in owner if Path(it["ref"]).exists()}
    clips = held_out_clips(root / "data/staging/vivid")
    items = []
    for it in owner:
        it = dict(it)
        it.setdefault("prompt", compact_prompt(it.get("category"), it.get("name"), it.get("description")))
        items.append(it)
    for k, pair in enumerate(pick_new_clips(clips, exclude, ref_shas), start=len(owner) + 1):
        pname, pmeta, pdir = pair["person"]
        dname, dmeta, ddir = pair["donor"]
        cat = APP_CATEGORY.get(dmeta.get("category", "upper"), "tops")
        desc = dmeta.get("garment_description") or None
        items.append({"key": f"{k}_heldout_{pname}", "case": "heldout", "video": str(pdir / "target.mp4"),
                      "ref": str(ddir / "ref.jpg"), "category": cat, "name": None, "description": desc,
                      "garment_from": dname, "prompt": compact_prompt(cat, None, desc)})
    out = root / "eval/fxa"
    out.mkdir(parents=True, exist_ok=True)
    # Owner-written garment descriptions (eval/fxa/descriptions.json: {"7": "...", "10": "..."} by item number
    # or by key); the prompt is rebuilt from them like the live app does, sleeve wording included.
    desc_file = out / "descriptions.json"
    if desc_file.exists():
        given = json.loads(desc_file.read_text())
        for i, it in enumerate(items, 1):
            text = given.get(str(i)) or given.get(it["key"])
            if text:
                it["description"] = text.strip()
                it["prompt"] = compact_prompt(it.get("category"), it.get("name"), it["description"])
                it["description_by"] = "owner"
    (out / "clips10.json").write_text(json.dumps(items, indent=2))
    for i, it in enumerate(items, 1):
        print(f"\n[{i}] {it['key']}  category={it['category']}  garment={Path(it['ref']).parent.name}/{Path(it['ref']).name}"
              f"\n    PROMPT: {it['prompt']}")
    print(f"\nwrote {out / 'clips10.json'} ({len(items)} clips)")
    return items


def reuse_renders(root: Path) -> dict:
    """Owner's 5 clips: the original and fxreal4 renders from eval/fx4, fxreal3 from eval/fx3 (same prompts,
    same seed: re-rendering would give the same frames)."""
    dst_root = root / "eval/fxa/results/clips10"
    sources = [(root / "eval/fx4/results/cases5", "out_base.mp4", "out_base.mp4"),
               (root / "eval/fx4/results/cases5", "out_f4_2000.mp4", "out_fx4.mp4"),
               (root / "eval/fx3/results/cases5", "out_f3_2000.mp4", "out_fx3.mp4")]
    copied, missing = 0, []
    for it in json.loads((root / "eval/fx2/cases5/clip_list.json").read_text()):
        dst = dst_root / it["key"]
        dst.mkdir(parents=True, exist_ok=True)
        base_dir = root / "eval/fx4/results/cases5" / it["key"]
        for extra in ("input.mp4", "item.json"):
            if (base_dir / extra).exists() and not (dst / extra).exists():
                shutil.copy(base_dir / extra, dst / extra)
        for ref in base_dir.glob("ref.*"):
            if not (dst / ref.name).exists():
                shutil.copy(ref, dst / ref.name)
        for src_dir, src_name, dst_name in sources:
            src = src_dir / it["key"] / src_name
            if src.exists():
                if not (dst / dst_name).exists():
                    shutil.copy(src, dst / dst_name)
                    copied += 1
            else:
                missing.append(str(src))
    report = {"copied": copied, "missing": missing}
    print(json.dumps(report, indent=1))
    return report


def used_clip_ids(root: Path) -> tuple[set, set]:
    """Clip ids (persons and garment donors) and garment-photo hashes used by every earlier clip list."""
    ids, shas = set(), set()
    for lst in list((root / "eval/fx2").glob("cases*/clip_list.json")) + list((root / "eval/fxa").glob("*.json")):
        try:
            items = json.loads(lst.read_text())
        except Exception:  # noqa: BLE001 - not a clip list (e.g. a descriptions file)
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("video"):
                ids.add(Path(it["video"]).parent.name)
            if it.get("garment_from"):
                ids.add(it["garment_from"])
            ref = Path(it["ref"]) if it.get("ref") else None
            if ref and ref.exists():
                shas.add(_sha(ref))
                if ref.parent.parent.name == "vivid":
                    ids.add(ref.parent.name)
    return ids, shas


def pick_set(root: Path, name: str, n: int = 10, seed: int = 11) -> list[dict]:
    """A new set of n held-out clips (tops, dresses, bottoms in turn), each with another held-out clip's
    garment of the same type; descriptions are left for the owner (NAME_descriptions.json)."""
    from data_tools.fluxrt_prompt import compact_prompt
    ids, shas = used_clip_ids(root)
    cats = tuple(("upper", "dress", "lower")[i % 3] for i in range(n))
    picked = pick_new_clips(held_out_clips(root / "data/staging/vivid"), ids, shas, categories=cats, seed=seed)
    items = []
    for k, pair in enumerate(picked, 1):
        pname, pmeta, pdir = pair["person"]
        dname, dmeta, ddir = pair["donor"]
        cat = APP_CATEGORY.get(dmeta.get("category", "upper"), "tops")
        items.append({"key": f"{k}_{name}_{pname}", "case": name, "video": str(pdir / "target.mp4"),
                      "ref": str(ddir / "ref.jpg"), "category": cat, "name": None, "description": None,
                      "garment_from": dname, "prompt": compact_prompt(cat, None, None), "needs_description": True})
    out = root / "eval/fxa"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(items, indent=2))
    tmpl = out / f"{name}_descriptions.json"
    if not tmpl.exists():
        tmpl.write_text(json.dumps({str(i): "" for i in range(1, len(items) + 1)}, indent=1))
    print(f"\n{len(items)} new clips (none used before). Look at each garment photo and describe it in {tmpl}:\n")
    for i, it in enumerate(items, 1):
        print(f"[{i}] {it['category']:8s} photo: {it['ref']}   (person clip {Path(it['video']).parent.name})")
    if len(items) < n:
        print(f"\nWARNING: only {len(items)} of {n} clips could be picked (not enough unused held-out clips of some type)")
    return items


def apply_descriptions(root: Path, name: str) -> list[dict]:
    """Every item must have an owner description; the prompt is rebuilt from it (sleeve wording included)."""
    from data_tools.fluxrt_prompt import compact_prompt
    out = root / "eval/fxa"
    items = json.loads((out / f"{name}.json").read_text())
    given = json.loads((out / f"{name}_descriptions.json").read_text()) if (out / f"{name}_descriptions.json").exists() else {}
    missing = []
    for i, it in enumerate(items, 1):
        text = (given.get(str(i)) or given.get(it["key"]) or "").strip()
        if len(text) < 8:
            missing.append(i)
            continue
        it["description"] = text
        it["prompt"] = compact_prompt(it.get("category"), it.get("name"), text)
        it["description_by"] = "owner"
        it.pop("needs_description", None)
    if missing:
        print(f"MISSING descriptions for items {missing} in {out / f'{name}_descriptions.json'}; nothing rendered")
        raise SystemExit(2)
    (out / f"{name}.json").write_text(json.dumps(items, indent=2))
    for i, it in enumerate(items, 1):
        print(f"\n[{i}] {it['key']}  category={it['category']}\n    PROMPT: {it['prompt']}")
    return items


def parse_panels(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        name, _, label = s.partition("=")
        out.append((name.strip(), (label or name).strip()))
    return out


def labelled_video(root: Path, panels: list[tuple[str, str]], repeats: int = 2, cols: int = 5,
                   height: int = 640, results: Path | None = None, out_path: Path | None = None,
                   set_name: str = "clips10", legend: bool = True) -> dict:
    import av
    from PIL import Image, ImageDraw
    from eval.owner_videos import _font, _frames, _panel
    res = results or root / "eval/fxa/results" / set_name
    items = json.loads((root / "eval/fxa" / f"{set_name}.json").read_text())
    out_path = out_path or root / "eval/fxa" / ("step_a_labelled.mp4" if set_name == "clips10" else f"{set_name}_labelled.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pw = height * 9 // 16 // 2 * 2
    n_panels = 2 + len(panels) + (1 if legend else 0)  # garment, camera, models, legend
    rows = -(-n_panels // cols)
    W, H = pw * cols, height * rows
    made, skipped = [], []
    with av.open(str(out_path), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "20", "preset": "medium"}
        for i, it in enumerate(items, 1):
            d = res / it["key"]
            paths = {name: d / f"out_{name}.mp4" for name, _ in panels}
            present = [(name, label) for name, label in panels if paths[name].exists()]
            if not (d / "input.mp4").exists() or not present:
                skipped.append(it["key"])
                continue
            cam = _frames(d / "input.mp4")
            vids = {name: _frames(paths[name]) for name, _ in present}
            ref = next((p for p in d.iterdir() if p.stem == "ref"), None)
            g = _panel(Image.open(ref).convert("RGB"), pw, height, f"{i}/{len(items)} garment") if ref else Image.new("RGB", (pw, height))
            n = min([len(cam)] + [len(v) for v in vids.values()])
            legend_tile = Image.new("RGB", (pw, height), (18, 18, 18))
            dr = ImageDraw.Draw(legend_tile)
            lines = [f"item {i} of {len(items)}", it.get("case", ""), f"type: {it.get('category')}", "",
                     "panels:", "garment, camera"] + [label for _, label in present] + ["", "(not blind)"]
            for k, line in enumerate(lines):
                dr.text((14, 50 + k * 34), line, fill=(230, 230, 230), font=_font(22))
            for _ in range(repeats):
                for k in range(n):
                    row = Image.new("RGB", (W, H), (0, 0, 0))
                    tiles = [g, _panel(cam[k], pw, height, "camera")]
                    tiles += [_panel(vids[name][k], pw, height, label) for name, label in present]
                    if legend:
                        tiles.append(legend_tile)
                    for t, tile in enumerate(tiles):
                        row.paste(tile, ((t % cols) * pw, (t // cols) * height))
                    for pkt in s.encode(av.VideoFrame.from_image(row)):
                        c.mux(pkt)
            made.append(it["key"])
        for pkt in s.encode():
            c.mux(pkt)
    report = {"video": str(out_path), "size": [W, H], "items": made, "skipped": skipped,
              "panels": [label for _, label in panels]}
    print(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("clips", "reuse", "video", "pick", "apply"):
        p = sub.add_parser(name)
        p.add_argument("root", nargs="?", default="/workspace/azyah-vton")
        if name in ("video", "pick", "apply"):
            p.add_argument("--set", default="clips10", help="clip set name (eval/fxa/<set>.json)")
        if name == "pick":
            p.add_argument("--n", type=int, default=10)
            p.add_argument("--seed", type=int, default=11)
        if name == "video":
            p.add_argument("--panels", nargs="+", default=DEFAULT_PANELS, help="file name=label, in order")
            p.add_argument("--repeats", type=int, default=2)
            p.add_argument("--cols", type=int, default=5)
            p.add_argument("--no-legend", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "clips":
        return build_clip_list(root)
    if a.cmd == "reuse":
        return reuse_renders(root)
    if a.cmd == "pick":
        return pick_set(root, a.set, a.n, a.seed)
    if a.cmd == "apply":
        return apply_descriptions(root, a.set)
    return labelled_video(root, parse_panels(a.panels), a.repeats, a.cols, set_name=a.set, legend=not a.no_legend)


if __name__ == "__main__":
    main()
