"""Add the face to each sample's loss-weight mask (fxreal3: face counts 2x, like the garment).

  python -m data_tools.face_weight DATASET_DIR [--model SEG.tflite]

DATASET_DIR/data/*.parquet come from build_real.py. For every row, MediaPipe's multiclass selfie
segmenter finds the face (class 3) in the answer image ("target"); the "mask" column becomes
max(garment mask, face area grown by 2 px). The training patch weights the mask area --mask_weight
times, so errors on the face now count as much as errors on the garment. Writes the files in place
(the originals are kept in DATASET_DIR/data_before_face/) and DATASET_DIR/face_weight.json. Needs pyarrow.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACE = 3


def merge_face(mask_png: bytes, target_jpg: bytes, seg) -> tuple[bytes, int]:
    import numpy as np
    from PIL import Image, ImageFilter
    target = Image.open(io.BytesIO(target_jpg)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_png)).convert("L").resize(target.size)
    face = Image.fromarray(((np.squeeze(seg(target)) == FACE) * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(5))
    out = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(face)), "L")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), int((np.asarray(face) > 0).sum())


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    a = ap.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_tools.clothes_mask import Segmenter
    seg = Segmenter(Path(a.model))
    root = Path(a.dataset)
    done = root / "face_weight.json"
    if done.exists():
        print(done.read_text())
        return json.loads(done.read_text())
    stats = {"rows": 0, "rows_with_face": 0, "files": 0}
    for f in sorted((root / "data").glob("*.parquet")):
        bak = root / "data_before_face" / f.name  # outside data/, so the trainer never reads it
        if not bak.exists():
            bak.parent.mkdir(exist_ok=True)
            shutil.copy(f, bak)
        t = pq.read_table(bak)
        masks, targets = t.column("mask").to_pylist(), t.column("target").to_pylist()
        new = []
        for m, tg in zip(masks, targets):
            b, n = merge_face(m["bytes"], tg["bytes"], seg)
            new.append({"bytes": b, "path": m["path"]})
            stats["rows"] += 1
            stats["rows_with_face"] += int(n > 30)
        i = t.schema.get_field_index("mask")
        t = t.set_column(i, t.schema.field(i), pa.array(new, type=t.schema.field(i).type))
        pq.write_table(t.replace_schema_metadata(t.schema.metadata), f)
        stats["files"] += 1
    done.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
