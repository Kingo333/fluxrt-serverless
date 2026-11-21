"""Loss weights for the face, skin and hair, and skin-tone balance (fxreal6; replaces face_weight there).

  python -m data_tools.region_weight DATASET_DIR [--model SEG.tflite] [--min-share 0.12] [--max-repeat 3]

DATASET_DIR/data/*.parquet come from build_real.py. For every row the answer picture ("target") is
segmented (MediaPipe multiclass selfie segmenter) and the "mask" column becomes the largest of:

  garment mask (as built)        full weight
  face (grown by 2 px)           full weight       -> with --mask_weight 3: garment 3x, face 3x
  neck, arms, hands, legs        half weight       ->                      skin 2x
  hair and head covering area    half weight       ->                      hair 2x

The trainer's weight is 1 + (mask_weight - 1) x mask, and its colour, texture and edge terms work on
the same area, so face and skin are held to the answer like the garment is. No trainer change.

Skin tone: each row's person gets an ITA skin-tone group (data_tools.skin_tone) from the answer's skin.
Rows of rare groups are repeated (both frames of a pair together, at most --max-repeat times) so no
group falls under --min-share. Writes in place (originals kept in DATASET_DIR/data_before_regions/)
and DATASET_DIR/region_weight.json. Needs pyarrow; runs in the MediaPipe venv.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_tools import skin_tone  # noqa: E402

HAIR, BODY_SKIN, FACE = 1, 2, 3
HALF = 128


def region_mask(mask, cat):
    """mask: garment weights, uint8 HxW; cat: category map of the answer, HxW. Returns the new uint8 mask."""
    import numpy as np
    from PIL import Image, ImageFilter
    face = np.asarray(Image.fromarray(((cat == FACE) * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(5)))
    soft = (np.isin(cat, (BODY_SKIN, HAIR)) * HALF).astype(np.uint8)
    return np.maximum(np.maximum(mask, face), soft)


def tone_group(rgb, cat, min_px: int = 150) -> str | None:
    """Skin-tone group of the person in one picture (face and body skin together), or None with too little skin."""
    import numpy as np
    from eval.garment_report import srgb_to_lab
    px = rgb[np.isin(cat, (FACE, BODY_SKIN))]
    if len(px) < min_px:
        return None
    return skin_tone.group(skin_tone.ita(np.median(srgb_to_lab(px), axis=0)))


def repeats(units: list[tuple[str, str | None]], min_share: float, max_repeat: int) -> dict[str, int]:
    """units: (unit id, skin-tone group or None) for every pair or single row. Returns how many times each
    unit is written (1 = once). Rare groups are repeated a whole number of times, never more than max_repeat."""
    groups = [g for _, g in units if g]
    w = skin_tone.balance_weights(groups, min_share=min_share, max_repeat=float(max_repeat))
    out = {}
    for uid, g in units:
        out[uid] = max(1, min(max_repeat, int(round(w.get(g, 1.0))))) if g else 1
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--min-share", type=float, default=0.12)
    ap.add_argument("--max-repeat", type=int, default=3)
    a = ap.parse_args(argv)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    seg = Segmenter(Path(a.model))
    root = Path(a.dataset)
    done = root / "region_weight.json"
    if done.exists():
        print(done.read_text())
        return json.loads(done.read_text())
    stats = {"rows": 0, "rows_with_face": 0, "rows_with_skin": 0, "files": 0}
    tables, units, unit_of_row = [], {}, []
    for f in sorted((root / "data").glob("*.parquet")):
        bak = root / "data_before_regions" / f.name  # outside data/, so the trainer never reads it
        if not bak.exists():
            bak.parent.mkdir(exist_ok=True)
            shutil.copy(f, bak)
        t = pq.read_table(bak)
        masks, targets = t.column("mask").to_pylist(), t.column("target").to_pylist()
        keys, pairs = t.column("key").to_pylist(), t.column("pair").to_pylist()
        new = []
        for m, tg, key, pair in zip(masks, targets, keys, pairs):
            target = Image.open(io.BytesIO(tg["bytes"])).convert("RGB")
            cat = np.squeeze(seg(target))
            mask = np.asarray(Image.open(io.BytesIO(m["bytes"])).convert("L").resize(target.size))
            buf = io.BytesIO()
            Image.fromarray(region_mask(mask, cat), "L").save(buf, format="PNG")
            new.append({"bytes": buf.getvalue(), "path": m["path"]})
            stats["rows"] += 1
            stats["rows_with_face"] += int((cat == FACE).sum() > 30)
            stats["rows_with_skin"] += int((cat == BODY_SKIN).sum() > 30)
            uid = f"pair:{pair}" if pair else f"row:{key}"
            units.setdefault(uid, tone_group(np.asarray(target), cat))
            unit_of_row.append(uid)
        i = t.schema.get_field_index("mask")
        tables.append((f, t.set_column(i, t.schema.field(i), pa.array(new, type=t.schema.field(i).type))))
    rep = repeats(list(units.items()), a.min_share, a.max_repeat)
    groups = [g for g in units.values() if g]
    stats["skin_tone_people"] = {g: groups.count(g) for g in skin_tone.GROUP_NAMES}
    stats["skin_tone_missing"] = skin_tone.missing(groups)
    stats["skin_tone_repeat"] = {g: max((rep[u] for u, gg in units.items() if gg == g), default=1) for g in skin_tone.GROUP_NAMES}
    row = 0
    for f, t in tables:
        n = t.num_rows
        times = [rep[unit_of_row[row + k]] for k in range(n)]
        row += n
        parts = [t]
        for extra in range(1, max(times, default=1)):
            idx = [k for k in range(n) if times[k] > extra]
            if not idx:
                continue
            more = t.take(pa.array(idx))
            suffix = f"_r{extra}"  # new key and pair id, so a repeated pair is still one pair
            for col in ("key", "pair"):
                j = more.schema.get_field_index(col)
                vals = [v + suffix if v else v for v in more.column(col).to_pylist()]
                more = more.set_column(j, more.schema.field(j), pa.array(vals, type=pa.string()))
            parts.append(more)
        out = pa.concat_tables(parts) if len(parts) > 1 else t
        pq.write_table(out.replace_schema_metadata(t.schema.metadata), f)
        stats["files"] += 1
        stats["rows_written"] = stats.get("rows_written", 0) + out.num_rows
    done.write_text(json.dumps(stats, indent=2))
    print("REGIONS " + json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
