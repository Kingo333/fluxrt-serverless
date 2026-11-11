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
from data_tools.vivid_motion import motion_items  # noqa: E402
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


def photo_jitter(img, brightness: float, contrast: float):
    """The same light change for the input and the answer (fxreal3): the model learns to keep a face
    and skin as they are in dim or bright light instead of 'correcting' them."""
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(ImageEnhance.Brightness(img.convert("RGB")).enhance(brightness)).enhance(contrast)


def jitter_factors(rng: random.Random, p: float):
    """(brightness, contrast) for a fraction p of samples, else None."""
    if p <= 0 or rng.random() >= p:
        return None
    return rng.uniform(0.6, 1.4), rng.uniform(0.75, 1.25)


def vivid_pair_items(root: Path):
    """data_tools.vivid_pairs output: (clip, category, ref, [(target_a, mask_a), (target_b, mask_b)])."""
    if not root or not root.is_dir():
        return
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mp = d / "meta.json"
        files = [d / f for f in ("a_target.png", "a_mask.png", "b_target.png", "b_mask.png", "ref.jpg")]
        if not mp.exists() or not all(f.exists() for f in files):
            continue
        meta = json.loads(mp.read_text())
        if meta.get("split", "train") != "train":
            continue
        yield d.name, meta.get("category", "upper"), d / "ref.jpg", [(files[0], files[1]), (files[2], files[3])]


def extend_down(mask, frac: float = 0.2, widen: float = 0.06):
    """fxreal4: the garment area plus a band below its hem, `frac` of the picture tall, as wide as
    the lowest part of the garment (+ `widen` each side). Returns (whole area, the added band only)
    as 'L' images. The re-dresser draws a LONGER garment there; the answer keeps the real, shorter
    one, so the model learns that leftover length turns back into trousers, skirt or skin."""
    import numpy as np
    from PIL import Image
    m = np.asarray(mask.convert("L")) > 127
    h, w = m.shape
    rows = np.where(m.any(axis=1))[0]
    band = np.zeros_like(m)
    if rows.size:
        bottom = int(rows.max())
        low = m[max(0, bottom - max(1, (bottom - int(rows.min())) // 10)): bottom + 1]
        cols = np.where(low.any(axis=0))[0]
        x0 = max(0, int(cols.min()) - int(widen * w))
        x1 = min(w, int(cols.max()) + 1 + int(widen * w))
        band[bottom + 1: min(h, bottom + 1 + int(frac * h)), x0:x1] = True
    whole = m | band
    return (Image.fromarray((whole * 255).astype(np.uint8), "L"),
            Image.fromarray((band * 255).astype(np.uint8), "L"))


LONGER_PROMPT = ("Virtual try-on. The person in the camera frame wears the garment from the reference image, but "
                 "made longer: a long tunic-length version that covers the hips and reaches the upper thighs, worn "
                 "over the trousers or skirt. Keep the person's face, hair, body, pose and the background unchanged.")


OUTFIT_PROMPT = ("Virtual try-on. The person in the camera frame wears the top from the reference image with plain, "
                 "full-length dark trousers instead of the dress. Keep the person's face, hair, body, pose and the "
                 "background unchanged.")


def caption_for(category, description) -> str:
    """The app's compact prompt, with the garment description when staging has one (fxreal5)."""
    return compact_prompt(category, None, description or None)


def meta_description(folder: Path):
    try:
        return json.loads((Path(folder) / "meta.json").read_text()).get("garment_description") or None
    except Exception:  # noqa: BLE001 - no meta, no description
        return None


def outfit_area(mask, widen: float = 0.06):
    """fxreal5 'top and trousers -> dress': the dress area plus everything below its hem to the bottom of
    the picture (the legs), where the re-dresser draws trousers. Returns (whole, below-hem band)."""
    return extend_down(mask, 1.0, widen)


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
                "pair": {"dtype": "string", "_type": "Value"},
                "person": feat, "garment": feat, "target": feat, "mask": feat}
    schema = pa.schema([("key", pa.string()), ("caption", pa.string()), ("pair", pa.string()), ("person", img),
                        ("garment", img), ("target", img), ("mask", img)],
                       metadata={"huggingface": json.dumps({"info": {"features": features}})})
    cols = {"key": [r["key"] for r in rows], "caption": [r["caption"] for r in rows],
            "pair": [r.get("pair", "") for r in rows]}
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
        self.prompts = {c: compact_prompt(c) for c in ("tops", "bottoms", "dresses")}
        self.embeds = {}
        self.cache_ok = True
        for c, text in self.prompts.items():
            try:
                enc = self.pipe.encode_prompt(prompt=text, device="cuda")
                self.embeds[c] = enc[0] if isinstance(enc, tuple) else enc
            except Exception as e:  # noqa: BLE001 - fall back to plain prompts
                print(f"(prompt cache off: {e})", flush=True)
                self.cache_ok = False
                break

    def __call__(self, person, garment, seed: int, category: str = "tops", prompt: str | None = None):
        from PIL import Image
        gw, gh = self.w * 2 // 3 // 16 * 16, self.h * 2 // 3 // 16 * 16
        g = garment.copy()
        g.thumbnail((gw, gh), Image.LANCZOS)
        canvas = Image.new("RGB", (gw, gh), (255, 255, 255))
        canvas.paste(g, ((gw - g.width) // 2, (gh - g.height) // 2))
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        kw = dict(image=[person.resize((self.w, self.h)), canvas], height=self.h, width=self.w,
                  num_inference_steps=self.steps, guidance_scale=1.0, generator=gen)
        if prompt:
            return self.pipe(prompt=prompt, **kw).images[0]
        cat = {"upper": "tops", "lower": "bottoms", "dress": "dresses"}.get(category, category)
        cat = cat if cat in self.prompts else "tops"
        if self.cache_ok and cat in self.embeds:
            try:
                return self.pipe(prompt_embeds=self.embeds[cat], **kw).images[0]
            except Exception as e:  # noqa: BLE001 - this diffusers version wants the text itself
                print(f"(prompt cache off: {type(e).__name__}: {e})", flush=True)
                self.cache_ok = False
        return self.pipe(prompt=self.prompts[cat], **kw).images[0]


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
    ap.add_argument("--vivid-pairs", help="data_tools.vivid_pairs output: two nearby frames per ViViD train clip "
                    "(fxreal3 same-clip colour term); both frames re-dressed with the same other garment")
    ap.add_argument("--photo-aug", type=float, default=0.0,
                    help="fraction of samples given the same random brightness/contrast change on input and answer")
    ap.add_argument("--longer-frac", type=float, default=0.0,
                    help="fxreal4: fraction of re-dressed tops whose INPUT garment is drawn longer than the answer's")
    ap.add_argument("--longer-ext", type=float, default=0.2, help="how far below the hem the longer garment may reach")
    ap.add_argument("--longer-min-change", type=float, default=0.06,
                    help="the band below the hem must change at least this much (0-1) or the sample stays a normal one")
    ap.add_argument("--motion", help="data_tools.vivid_motion output (fxreal5): dress frames where the dress moves")
    ap.add_argument("--dress-outfit-frac", type=float, default=0.0,
                    help="fxreal5: share of motion clips whose INPUT is a top and trousers (the live case) instead of another dress")
    ap.add_argument("--other-live-max", type=int, default=-1,
                    help="cap on non-dress ViViD live stills (-1 = no cap); fxreal5 keeps dresses the bulk")
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
                                 "vivid_live": 0, "vivid_no_mask": 0, "vivid_misaligned": 0, "test_pairs": 0,
                                 "pair_clips_tried": 0, "pair_clips_kept": 0, "photo_aug": 0,
                                 "longer_tried": 0, "longer_kept": 0, "longer_pairs": 0,
                                 "vivid_live_dress": 0, "vivid_live_other_capped": 0, "described": 0,
                                 "motion_clips_tried": 0, "motion_pairs_kept": 0, "motion_singles_kept": 0,
                                 "motion_outfit_kept": 0}
    used = set()

    def flush(final=False):
        nonlocal rows, files
        if rows and (final or len(rows) >= a.per_file):
            write_parquet(rows, out / "data" / f"train-{files:05d}.parquet")
            files, rows = files + 1, []

    def add(key, caption, person, garment, target, mask, pair="", jitter="auto"):
        g = garment.copy()
        if max(g.size) > 1024:
            g.thumbnail((1024, 1024), Image.LANCZOS)
        jf = jitter_factors(rng, a.photo_aug) if jitter == "auto" else jitter
        if jf:
            person, target = photo_jitter(person, *jf), photo_jitter(target, *jf)
            stats["photo_aug"] += 1
        rows.append({"key": key, "caption": caption, "pair": pair, "person": _jpeg(_cover(person, w, h)),
                     "garment": _jpeg(g), "target": _jpeg(_cover(target, w, h)),
                     "mask": _png(_cover(mask.convert("L"), w, h))})
        flush()

    # 1. ViViD live-way stills (no GPU): real frame pasted back outside the garment area
    for key, cat, p, g, t in vivid_live_items([Path(d) for d in a.vivid_stills]):
        mpath = p.parent / "garment_mask.png"
        if not mpath.exists():
            stats["vivid_no_mask"] += 1
            continue
        if cat != "dress" and a.other_live_max >= 0 and stats["vivid_live"] - stats["vivid_live_dress"] >= a.other_live_max:
            stats["vivid_live_other_capped"] += 1
            continue
        mask = Image.open(mpath).convert("L")
        real = _cover(Image.open(t).convert("RGB"), *mask.size)
        live = _cover(Image.open(p).convert("RGB"), *mask.size)
        soft = soft_mask(mask)
        if outside_diff(live, real, soft) > a.max_outside_diff:
            stats["vivid_misaligned"] += 1
            continue
        desc = meta_description(p.parent)
        stats["described"] += int(bool(desc))
        add(key, caption_for(cat, desc), paste_back(live, real, soft), Image.open(g).convert("RGB"), real, soft)
        used.add("vivid")
        stats["vivid_live"] += 1
        stats["vivid_live_dress"] += int(cat == "dress")
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
        longer = a.longer_frac > 0 and rng.random() < a.longer_frac
        gen = redress(real, cloth_j, seed=a.seed * 100003 + i, **({"prompt": LONGER_PROMPT} if longer else {}))
        soft = soft_mask(mask)
        inp = paste_back(gen, real, soft)
        tag = ""
        if longer:
            stats["longer_tried"] += 1
            whole, band = extend_down(mask, a.longer_ext)
            inp_long = paste_back(gen, real, soft_mask(whole))
            if changed_enough(inp_long, real, band, a.longer_min_change):
                inp, soft, mask, tag = inp_long, soft_mask(whole), whole, "_long"
        bb = bbox_of(mask)
        sim = dino(inp.crop(bb), real.crop(bb)) if bb else 1.0
        if not bb or sim > a.max_sim or not changed_enough(inp, real, mask):
            stats["viton_unchanged"] += 1
            continue
        add(f"viton_hd_{i:05d}{tag}", caption, inp, cloth_i, real, soft)
        used.add("viton_hd")
        stats["viton_kept"] += 1
        stats["longer_kept"] += int(bool(tag))
        if (n + 1) % 100 == 0:
            print(json.dumps({"progress": n + 1, **stats}), flush=True)

    # 3. ViViD same-clip pairs (fxreal3): two nearby real frames, both re-dressed with the SAME other
    #    garment and seed and pasted back; kept or dropped together. The training patch keeps the two
    #    in one batch and asks their garment colour to differ only as much as the real frames do.
    pairs = list(vivid_pair_items(Path(a.vivid_pairs))) if a.vivid_pairs else []
    by_cat: dict = {}
    for clip, cat, ref, _ in pairs:
        by_cat.setdefault(cat, []).append(ref)
    for clip, cat, ref, frames in pairs:
        stats["pair_clips_tried"] += 1
        others = [r for r in by_cat.get(cat, []) if r != ref]
        if not others:
            continue
        other = Image.open(rng.choice(others)).convert("RGB")
        seed = a.seed * 100003 + 7919 + stats["pair_clips_tried"]
        built = []
        longer = cat == "upper" and a.longer_frac > 0 and rng.random() < a.longer_frac
        for tpath, mpath in frames:
            real = _cover(Image.open(tpath).convert("RGB"), gw, gh)
            mask = Image.open(mpath).convert("L").resize((gw, gh))
            gen = redress(real, other, seed=seed, category=cat, **({"prompt": LONGER_PROMPT} if longer else {}))
            if longer:
                whole, band = extend_down(mask, a.longer_ext)
                soft = soft_mask(whole, grow=13)
                inp = paste_back(gen, real, soft)
                if not changed_enough(inp, real, band, a.longer_min_change):
                    break
            else:
                soft = soft_mask(mask, grow=13)
                inp = paste_back(gen, real, soft)
            bb = bbox_of(mask)
            sim = dino(inp.crop(bb), real.crop(bb)) if bb else 1.0
            if not bb or sim > a.max_sim or not changed_enough(inp, real, mask):
                break
            built.append((inp, real, soft))
        if len(built) != 2:
            continue
        jf = jitter_factors(rng, a.photo_aug)
        garment = Image.open(ref).convert("RGB")
        for tag, (inp, real, soft) in zip("ab", built):
            add(f"vivid_pair_{clip}_{tag}", compact_prompt(cat), inp, garment, real, soft, pair=clip, jitter=jf)
        used.add("vivid")
        stats["pair_clips_kept"] += 1
        stats["pair_clips_kept_dress"] = stats.get("pair_clips_kept_dress", 0) + int(cat == "dress")
        stats["longer_pairs"] += int(longer)
    print(json.dumps({"pair_clips_kept": stats["pair_clips_kept"], "pair_clips_tried": stats["pair_clips_tried"]}),
          flush=True)

    # 4. fxreal5: dress frames where the dress MOVES (data_tools.vivid_motion). The pair (a, b) is re-dressed
    #    with the same garment and seed (pair term); singles add more poses of the same dress. A share of
    #    clips get a top-and-trousers INPUT (the live case: someone in a T-shirt and trousers tries an abaya),
    #    the rest another dress. The answer is always the real frame.
    motion = list(motion_items(Path(a.motion))) if a.motion else []
    dress_refs = [m[2] for m in motion]
    for clip, cat, ref, desc, pair, singles in motion:
        stats["motion_clips_tried"] += 1
        outfit = a.dress_outfit_frac > 0 and rng.random() < a.dress_outfit_frac
        if outfit:
            other = bank(rng.choice(train_pool))
        else:
            others = [r for r in dress_refs if r != ref]
            if not others:
                continue
            other = Image.open(rng.choice(others)).convert("RGB")
        seed = a.seed * 100003 + 15485 + stats["motion_clips_tried"]
        garment = Image.open(ref).convert("RGB")
        caption = caption_for(cat, desc)

        def make(tpath, mpath):
            real = _cover(Image.open(tpath).convert("RGB"), gw, gh)
            mask = Image.open(mpath).convert("L").resize((gw, gh))
            if outfit:
                whole, band = outfit_area(mask)
                gen = redress(real, other, seed=seed, prompt=OUTFIT_PROMPT)
                soft = soft_mask(whole, grow=13)
                inp = paste_back(gen, real, soft)
                if band.getbbox() and not changed_enough(inp, real, band, a.longer_min_change):
                    return None
                area = whole
            else:
                gen = redress(real, other, seed=seed, category=cat)
                soft = soft_mask(mask, grow=13)
                inp = paste_back(gen, real, soft)
                area = mask
            bb = bbox_of(mask)
            sim = dino(inp.crop(bb), real.crop(bb)) if bb else 1.0
            if not bb or sim > a.max_sim or not changed_enough(inp, real, area):
                return None
            return inp, real, soft

        built = [make(*f) for f in pair]
        if all(built):
            jf = jitter_factors(rng, a.photo_aug)
            for tag, (inp, real, soft) in zip("ab", built):
                add(f"vivid_motion_{clip}_{tag}", caption, inp, garment, real, soft, pair=f"m_{clip}", jitter=jf)
            stats["motion_pairs_kept"] += 1
            stats["motion_outfit_kept"] += int(outfit)
            stats["described"] += 2 * int(bool(desc))
        for k, f in enumerate(singles, 1):
            got = make(*f)
            if got:
                add(f"vivid_motion_{clip}_s{k}", caption, *got)
                stats["motion_singles_kept"] += 1
                stats["described"] += int(bool(desc))
        used.add("vivid")
        if stats["motion_clips_tried"] % 50 == 0:
            print(json.dumps({"motion_progress": stats["motion_clips_tried"], "pairs": stats["motion_pairs_kept"],
                              "singles": stats["motion_singles_kept"]}), flush=True)
    if motion:
        print(json.dumps({k: stats[k] for k in ("motion_clips_tried", "motion_pairs_kept", "motion_singles_kept",
                                                 "motion_outfit_kept")}), flush=True)
    flush(final=True)

    sel = selection_for(used)
    sel["owner_accepted_risk"] = "2026-11-08: research-licence real data accepted for R&D; contact rights holders before commercial use"
    (out / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    total = (stats["vivid_live"] + stats["viton_kept"] + stats["nochange"] + 2 * stats["pair_clips_kept"]
             + 2 * stats["motion_pairs_kept"] + stats["motion_singles_kept"])
    dress_rows = (stats["vivid_live_dress"] + 2 * stats.get("pair_clips_kept_dress", 0)
                  + 2 * stats["motion_pairs_kept"] + stats["motion_singles_kept"])
    stats["dress_rows"] = dress_rows
    stats["dress_share"] = round(dress_rows / total, 3) if total else 0.0
    summary = {"samples": total, "files": files, "size": [w, h], "dataset_name": str(out / "data"), **stats}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
