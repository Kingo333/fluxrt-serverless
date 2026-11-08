"""Garment-area masks for ViViD stills with a live-way input (owner, 2026-11-08: the LoRA may change
ONLY the garment; face, skin and background must stay the real pixels).

  python -m data_tools.clothes_mask S/vivid_dress_stills S/vivid_upper_stills S/vivid_lower_stills [--size 432x768]

The live-way inputs were made by JoyAI, which redraws the whole frame, so outside the garment they
are not the real pixels. For each item with live_source.png this writes garment_mask.png: the
union of "clothes" and "others" (bags, belts) in BOTH the real frame and the live input, as
segmented by MediaPipe's multiclass selfie segmenter (Apache-2.0), on both pictures centre-cropped
to portrait --size. fluxrt_lora.build_real then pastes the real frame back everywhere outside it.
Needs MediaPipe (run with the venv-mp Python). Resumable.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
             "float32/latest/selfie_multiclass_256x256.tflite")
CLOTHES, OTHERS = 4, 5
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def cover(img, w: int, h: int):
    from PIL import Image
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def union_mask(cat_a, cat_b, classes=(CLOTHES, OTHERS)):
    """255 where either category map has one of the classes (uint8 arrays of equal shape)."""
    import numpy as np
    m = np.isin(cat_a, classes) | np.isin(cat_b, classes)
    return (m * 255).astype("uint8")


class Segmenter:
    def __init__(self, model: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        if not model.exists():
            urllib.request.urlretrieve(MODEL_URL, model)
        self.mp = mp
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)), output_category_mask=True))

    def __call__(self, img):
        import numpy as np
        r = self.seg.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.asarray(img.convert("RGB"))))
        return r.category_mask.numpy_view().copy()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--size", default="432x768")
    ap.add_argument("--model", default=str(Path.home() / "selfie_multiclass_256x256.tflite"))
    a = ap.parse_args(argv)
    from PIL import Image
    w, h = (int(v) for v in a.size.lower().split("x"))
    seg = Segmenter(Path(a.model))
    stats = {"written": 0, "kept": 0, "no_input": 0}
    for root in map(Path, a.dirs):
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / "garment_mask.png").exists():
                stats["kept"] += 1
                continue
            tgt = next((d / f"target{e}" for e in IMG_EXT if (d / f"target{e}").exists()), None)
            if not (tgt and (d / "live_source.png").exists()):
                stats["no_input"] += 1
                continue
            t = cover(Image.open(tgt).convert("RGB"), w, h)
            s = cover(Image.open(d / "live_source.png").convert("RGB"), w, h)
            Image.fromarray(union_mask(seg(t), seg(s)), "L").save(d / "garment_mask.png")
            stats["written"] += 1
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
