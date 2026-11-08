"""FluxRT LoRA dataset from REAL photos and video frames only (owner, 2026-11-08).

  python fluxrt_lora/build_real.py --out DATA_DIR --viton-dir RAW/viton_hd --klein CKPT/FLUX.2-klein-4B \
      --vivid-stills S/vivid_dress_stills S/vivid_upper_stills S/vivid_lower_stills --test-out EVAL/viton_pairs

Every training example keeps a REAL picture as the answer (target); a model-made picture only
ever appears on the input side (person):

- VITON-HD (forgeml/viton_hd, real studio photos of models in tops + flat garment photos;
  research licence, owner accepted the risk): target = the real photo, garment = its real flat
  photo, person = the same photo re-dressed in ANOTHER VITON-HD top by FLUX.2-klein-4B
  (Apache-2.0, 4 steps), then everything outside the garment area is pasted back from the real
  photo (the area comes from VITON-HD's own clothing-agnostic image), so input and answer are
  identical outside the garment. Inputs where the garment did not visibly change are dropped
  (DINOv2-small similarity + pixel change inside the area).
- ViViD real frames that already have a live-way input (step 3b: the live JoyAI 0811 clone
  re-dressed the real frame with a fixed garment photo; dresses, tops, bottoms). JoyAI redraws the
  whole frame, so the real frame is pasted back everywhere outside the garment area
  (garment_mask.png from data_tools.clothes_mask); items without a mask, or whose input does not
  line up with the real frame outside the garment, are dropped.
- "No change" examples (--nochange-frac of the VITON-HD count): the person already wears the
  reference garment, so the answer is the input. Teaches "change nothing else".

All pictures are centre-cropped to portrait and resized to --size (288x512, FluxRT's frame);
the garment stays whole (the trainer pads it to --cond_2_size). Captions are the app's compact
FluxRT prompt (compact_prompt). Every row also carries a "mask" image (the garment area, white)
for the trainer's --mask_column: errors there count double, everywhere else (face, skin,
background: input == answer) 1x, so the LoRA learns to change only the garment. Held-out VITON-HD rows (the last --holdout rows, never
trained on) become eval pairs in --test-out (person.png = real photo, ref.png = another
held-out garment, meta.json) for eval.generate_fluxrt --pairs.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools.fluxrt_prompt import compact_prompt  # noqa: E402
from fluxrt_lora.make_dataset import _cover, _jpeg, selection_for  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def portrait_box(w: int, h: int, aspect: float = 9 / 16) -> tuple[int, int, int, int]:
    """Largest centred box of the given width/height aspect inside a w x h image."""
    if w / h > aspect:
        cw = round(h * aspect)
        x0 = (w - cw) // 2
        return x0, 0, x0 + cw, h
    ch = round(w / aspect)
    y0 = (h - ch) // 2
    return 0, y0, w, y0 + ch


def agnostic_mask(person, agnostic, threshold: int = 40):
    """Where VITON-HD's clothing-agnostic image differs from the photo (its grey garment+arms
    area): an 'L' image, 255 = may change."""
    import numpy as np
    from PIL import Image
    a = np.asarray(person.convert("RGB"), dtype=np.int16)
    b = np.asarray(agnostic.convert("RGB").resize(person.size), dtype=np.int16)
    m = (np.abs(a - b).sum(axis=2) > threshold).astype(np.uint8) * 255
    return Image.fromarray(m, "L")


def soft_mask(mask, grow: int = 9, feather: int = 4):
    """Grow the area a little (seams fall inside it) and soften its edge."""
    from PIL import ImageFilter
    m = mask.filter(ImageFilter.MaxFilter(grow * 2 + 1)) if grow > 0 else mask
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else m


def paste_back(generated, real, mask):
    """generated inside the (soft) mask, the real photo everywhere else."""
    from PIL import Image
    return Image.composite(generated.convert("RGB").resize(real.size), real.convert("RGB"), mask.resize(real.size))


def changed_enough(inp, target, mask, min_mean_diff: float = 0.035) -> bool:
    """Mean absolute pixel change inside the mask (0-1 scale) is at least min_mean_diff."""
    import numpy as np
    m = np.asarray(mask.resize(target.size), dtype=np.float32) / 255.0
    if m.sum() < 1:
        return False
    d = np.abs(np.asarray(inp.convert("RGB"), dtype=np.float32) - np.asarray(target.convert("RGB"), dtype=np.float32)).mean(axis=2) / 255.0
    return float((d * m).sum() / m.sum()) >= min_mean_diff


def bbox_of(mask, pad: int = 8):
    box = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if not box:
        return None
    x0, y0, x1, y1 = box
    w, h = mask.size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def pick_other(i: int, pool: list[int], rng: random.Random) -> int:
    j = i
    while j == i and len(pool) > 1:
        j = rng.choice(pool)
    return j


def outside_diff(inp, target, mask) -> float:
    """Mean absolute difference (0-1) OUTSIDE the mask: how well a model-made input lines up."""
    import numpy as np
    m = 1.0 - np.asarray(mask.resize(target.size), dtype=np.float32) / 255.0
    if m.sum() < 1:
        return 1.0
    d = np.abs(np.asarray(inp.convert("RGB").resize(target.size), dtype=np.float32)
               - np.asarray(target.convert("RGB"), dtype=np.float32)).mean(axis=2) / 255.0
    return float((d * m).sum() / m.sum())


def write_parquet(rows: list[dict], path: Path) -> None:
    """person / garment / target / mask images + caption, in the Hugging Face datasets layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    img = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    feat = {"_type": "Image"}
    features = {"key": {"dtype": "string", "_type": "Value"}, "caption": {"dtype": "string", "_type": "Value"},
                "person": feat, "garment": feat, "target": feat, "mask": feat}
    schema = pa.schema([("key", pa.string()), ("caption", pa.string()), ("person", img), ("garment", img),
                        ("target", img), ("mask", img)],
                       metadata={"huggingface": json.dumps({"info": {"features": features}})})
    cols = {"key": [r["key"] for r in rows], "caption": [r["caption"] for r in rows]}
    for c in ("person", "garment", "target", "mask"):
        ext = "png" if c == "mask" else "jpg"
        cols[c] = [{"bytes": r[c], "path": f"{r['key']}_{c}.{ext}"} for r in rows]
    pq.write_table(pa.table(cols, schema=schema), path)


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- data access
def viton_rows(viton_dir: Path):
    """Lazily yields (cloth, image, agnostic) PIL images by global row index from the parquet files."""
    import pyarrow.parquet as pq
    files = sorted(viton_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {viton_dir}")
    index = []
    for f in files:
        n = pq.ParquetFile(f).metadata.num_rows
        index += [(f, k) for k in range(n)]
    return index


class ClothBank:
    """Every VITON-HD garment photo (bytes), read once per parquet file: the re-dress garment comes
    from a random other row, which must not re-read a whole file each time."""

    def __init__(self, index):
        self.index, self.cache = index, {}

    def __call__(self, i: int):
        from PIL import Image
        import pyarrow.parquet as pq
        f, k = self.index[i]
        if f not in self.cache:
            self.cache[f] = pq.read_table(f, columns=["cloth"]).column("cloth").to_pylist()
        return Image.open(io.BytesIO(self.cache[f][k]["bytes"])).convert("RGB")


def load_viton(entry, cache: dict):
    from PIL import Image
    import pyarrow.parquet as pq
    f, k = entry
    if cache.get("file") != f:
        cache["file"], cache["table"] = f, pq.read_table(f, columns=["cloth", "image", "agnostic"])
    t = cache["table"]
    get = lambda c: Image.open(io.BytesIO(t.column(c)[k].as_py()["bytes"])).convert("RGB")  # noqa: E731
    return get("cloth"), get("image"), get("agnostic")


def vivid_live_items(dirs: list[Path]):
    """Step 3b ViViD stills that have a live-way input: (key, category, person, garment, target paths)."""
    for root in dirs:
        if not root.is_dir():
            print(f"(skip: {root} not found)", flush=True)
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            mp = d / "meta.json"
            if not mp.exists() or not (d / "live_source.png").exists():
                continue
            meta = json.loads(mp.read_text())
            if meta.get("split", "train") != "train" or not meta.get("live_source"):
                continue
            tgt = next((d / f"target{e}" for e in IMG_EXT if (d / f"target{e}").exists()), None)
            ref = next((p for p in d.iterdir() if p.stem == "ref" and p.suffix.lower() in IMG_EXT), None)
            if tgt and ref:
                yield f"vivid_{root.name}_{d.name}", meta.get("category", "upper"), d / "live_source.png", ref, tgt


# ---------------------------------------------------------------- models
class Redresser:
    """FLUX.2-klein-4B (distilled, 4 steps) re-dresses the person in another garment."""

    def __init__(self, klein_dir: str, gen_size: tuple[int, int], steps: int):
        import torch
        from diffusers import Flux2KleinPipeline
        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(klein_dir, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.w, self.h = gen_size
        self.steps = steps
        self.prompt = compact_prompt("tops")
        self.embeds = None
        try:
            enc = self.pipe.encode_prompt(prompt=self.prompt, device="cuda")
            self.embeds = enc[0] if isinstance(enc, tuple) else enc
        except Exception as e:  # noqa: BLE001 - fall back to plain prompts
            print(f"(prompt cache off: {e})", flush=True)

    def __call__(self, person, garment, seed: int):
        from PIL import Image
        gw, gh = self.w * 2 // 3 // 16 * 16, self.h * 2 // 3 // 16 * 16
        g = garment.copy()
        g.thumbnail((gw, gh), Image.LANCZOS)
        canvas = Image.new("RGB", (gw, gh), (255, 255, 255))
        canvas.paste(g, ((gw - g.width) // 2, (gh - g.height) // 2))
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        kw = dict(image=[person.resize((self.w, self.h)), canvas], height=self.h, width=self.w,
                  num_inference_steps=self.steps, guidance_scale=1.0, generator=gen)
        if self.embeds is not None:
            try:
                return self.pipe(prompt_embeds=self.embeds, **kw).images[0]
            except Exception as e:  # noqa: BLE001 - this diffusers version wants the text itself
                print(f"(prompt cache off: {type(e).__name__}: {e})", flush=True)
                self.embeds = None
        return self.pipe(prompt=self.prompt, **kw).images[0]


class DinoSim:
    """DINOv2-small (Apache-2.0) cosine similarity of two crops."""

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()

    def __call__(self, a, b) -> float:
        with self.torch.no_grad():
            x = self.proc(images=[a, b], return_tensors="pt").to("cuda")
            f = self.model(**x).pooler_output
            f = self.torch.nn.functional.normalize(f, dim=-1)
            return float((f[0] * f[1]).sum())


# ---------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viton-dir", required=True)
    ap.add_argument("--klein", required=True, help="FLUX.2-klein-4B folder (re-dressing the inputs)")
    ap.add_argument("--vivid-stills", nargs="*", default=[])
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--viton-n", type=int, default=2000, help="VITON-HD training examples to try")
    ap.add_argument("--holdout", type=int, default=300, help="last VITON-HD rows kept out of training")
    ap.add_argument("--test-items", type=int, default=50)
    ap.add_argument("--nochange-frac", type=float, default=0.1)
    ap.add_argument("--size", default="288x512")
    ap.add_argument("--gen-size", default="432x768")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-sim", type=float, default=0.93, help="drop inputs whose garment crop is this similar to the answer")
    ap.add_argument("--max-outside-diff", type=float, default=0.12,
                    help="drop ViViD live inputs that differ more than this (0-1) from the real frame outside the "
                         "garment area (misaligned; pasting back would leave seams)")
    ap.add_argument("--per-file", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from PIL import Image

    rng = random.Random(a.seed)
    w, h = (int(v) for v in a.size.lower().split("x"))
    gw, gh = (int(v) for v in a.gen_size.lower().split("x"))
    out, test_out = Path(a.out), Path(a.test_out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    rows, files, stats = [], 0, {"viton_tried": 0, "viton_kept": 0, "viton_unchanged": 0, "nochange": 0,
                                 "vivid_live": 0, "vivid_no_mask": 0, "vivid_misaligned": 0, "test_pairs": 0}
    used = set()

    def flush(final=False):
        nonlocal rows, files
        if rows and (final or len(rows) >= a.per_file):
            write_parquet(rows, out / "data" / f"train-{files:05d}.parquet")
            files, rows = files + 1, []

    def add(key, caption, person, garment, target, mask):
        g = garment.copy()
        if max(g.size) > 1024:
            g.thumbnail((1024, 1024), Image.LANCZOS)
        rows.append({"key": key, "caption": caption, "person": _jpeg(_cover(person, w, h)),
                     "garment": _jpeg(g), "target": _jpeg(_cover(target, w, h)),
                     "mask": _png(_cover(mask.convert("L"), w, h))})
        flush()

    # 1. ViViD live-way stills (no GPU): real frame pasted back outside the garment area
    for key, cat, p, g, t in vivid_live_items([Path(d) for d in a.vivid_stills]):
        mpath = p.parent / "garment_mask.png"
        if not mpath.exists():
            stats["vivid_no_mask"] += 1
            continue
        mask = Image.open(mpath).convert("L")
        real = _cover(Image.open(t).convert("RGB"), *mask.size)
        live = _cover(Image.open(p).convert("RGB"), *mask.size)
        soft = soft_mask(mask)
        if outside_diff(live, real, soft) > a.max_outside_diff:
            stats["vivid_misaligned"] += 1
            continue
        add(key, compact_prompt(cat), paste_back(live, real, soft), Image.open(g).convert("RGB"), real, soft)
        used.add("vivid")
        stats["vivid_live"] += 1
    print(json.dumps({"vivid_live": stats["vivid_live"]}), flush=True)

    # 2. VITON-HD: held-out eval pairs, then re-dressed training examples
    index = viton_rows(Path(a.viton_dir))
    holdout = list(range(max(0, len(index) - a.holdout), len(index)))
    train_pool = list(range(0, max(0, len(index) - a.holdout)))
    cache: dict = {}
    bank = ClothBank(index)
    for k, i in enumerate(holdout[: a.test_items]):
        d = test_out / f"viton_{i:05d}"
        if (d / "meta.json").exists():
            continue
        j = holdout[(k + len(holdout) // 2) % len(holdout)]
        _, img, _ = load_viton(index[i], cache)
        cloth_j = bank(j)
        d.mkdir(parents=True, exist_ok=True)
        img.crop(portrait_box(*img.size)).save(d / "person.png")
        cloth_j.save(d / "ref.png")
        (d / "meta.json").write_text(json.dumps({"category": "tops", "split": "test", "source": "viton_hd",
                                                 "person_row": i, "garment_row": j}))
        stats["test_pairs"] += 1

    picks = sorted(rng.sample(train_pool, min(a.viton_n, len(train_pool))))
    n_nochange = int(round(a.nochange_frac * len(picks)))
    nochange = set(rng.sample(picks, n_nochange)) if n_nochange else set()
    redress = Redresser(a.klein, (gw, gh), a.steps)
    dino = DinoSim()
    caption = compact_prompt("upper")
    for n, i in enumerate(picks):
        cloth_i, img, agn = load_viton(index[i], cache)
        box = portrait_box(*img.size)
        real = img.crop(box).resize((gw, gh), Image.LANCZOS)
        if i in nochange:
            same_mask = soft_mask(agnostic_mask(img, agn).crop(box).resize((gw, gh)))
            add(f"viton_hd_{i:05d}_same", caption, real, cloth_i, real, same_mask)
            stats["nochange"] += 1
            used.add("viton_hd")
            continue
        stats["viton_tried"] += 1
        mask = agnostic_mask(img, agn).crop(box).resize((gw, gh))
        j = pick_other(i, train_pool, rng)
        cloth_j = bank(j)
        gen = redress(real, cloth_j, seed=a.seed * 100003 + i)
        soft = soft_mask(mask)
        inp = paste_back(gen, real, soft)
        bb = bbox_of(mask)
        sim = dino(inp.crop(bb), real.crop(bb)) if bb else 1.0
        if not bb or sim > a.max_sim or not changed_enough(inp, real, mask):
            stats["viton_unchanged"] += 1
            continue
        add(f"viton_hd_{i:05d}", caption, inp, cloth_i, real, soft)
        used.add("viton_hd")
        stats["viton_kept"] += 1
        if (n + 1) % 100 == 0:
            print(json.dumps({"progress": n + 1, **stats}), flush=True)
    flush(final=True)

    sel = selection_for(used)
    sel["owner_accepted_risk"] = "2026-11-08: research-licence real data accepted for R&D; contact rights holders before commercial use"
    (out / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    total = stats["vivid_live"] + stats["viton_kept"] + stats["nochange"]
    summary = {"samples": total, "files": files, "size": [w, h], "dataset_name": str(out / "data"), **stats}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
