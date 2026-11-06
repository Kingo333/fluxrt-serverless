"""Person-part masks for FluxRT inference-time fixes (owner, 2026-11-06: improve live FluxRT in hours,
without training).

1. Real face/background composite: keep the camera frame's face, hair and background and take only
   the clothing area from the generated frame, so the person and the room never drift.
2. Clothing-agnostic input for the xocialize try-on LoRA (outfit region greyed out).

Segmenters:
- MediaPipe multiclass selfie segmenter (Apache-2.0, Google): background, hair, body skin, face
  skin, clothes, others. Commercial-OK; used for the composite (shippable).
- SegFormer B2 clothes (mattmdjaga/segformer_b2_clothes, licence "other", trained on the ATR
  research dataset): upper/skirt/pants/dress split, needed for the LoRA's top/bottom references.
  OFFLINE A/B ONLY until the licence is cleared or a commercial-OK parser replaces it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
                "float32/latest/selfie_multiclass_256x256.tflite")
MP = {"background": 0, "hair": 1, "body_skin": 2, "face": 3, "clothes": 4, "others": 5}
SF = {"background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper": 4, "skirt": 5, "pants": 6, "dress": 7,
      "belt": 8, "lshoe": 9, "rshoe": 10, "face": 11, "lleg": 12, "rleg": 13, "larm": 14, "rarm": 15, "bag": 16,
      "scarf": 17}
OUTFIT_SF = (SF["upper"], SF["skirt"], SF["pants"], SF["dress"], SF["belt"], SF["scarf"])
GARMENT_CATS = {"upper", "lower", "dress", "tops", "bottoms", "dresses", "outerwear"}


class MediaPipeSegmenter:
    """Category map per image (H x W uint8, classes in MP)."""

    def __init__(self, model_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        path = Path(model_path or Path.home() / ".cache" / "selfie_multiclass_256x256.tflite")
        if not path.exists():
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MP_MODEL_URL, path)
        self._mp = mp
        opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(path)),
                                            output_category_mask=True)
        self.seg = vision.ImageSegmenter.create_from_options(opts)

    def __call__(self, img) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
        res = self.seg.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
        return res.category_mask.numpy_view().copy()


class SegformerParser:
    """Category map per image (H x W uint8, classes in SF). Offline A/B only (licence)."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        self.torch, self.device = torch, device
        self.proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device).eval()

    def __call__(self, img) -> np.ndarray:
        torch = self.torch
        inputs = self.proc(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)


def keep_mask(cat_in: np.ndarray, cat_out: np.ndarray, category: str | None) -> np.ndarray:
    """Pixels to take from the REAL camera frame (MediaPipe classes). Garments: background, hair and
    face, minus anything that is clothing in the generated frame (a longer dress may cover former
    background). Shoes, bags, accessories: face and hair only (no shoe masks; owner 2026-11-06)."""
    face_hair = np.isin(cat_in, (MP["hair"], MP["face"]))
    if str(category or "").lower() in GARMENT_CATS:
        keep = face_hair | (cat_in == MP["background"])
        return keep & ~np.isin(cat_out, (MP["clothes"], MP["others"]))
    return face_hair & ~np.isin(cat_out, (MP["clothes"],))


def composite(generated, real, keep: np.ndarray, feather: float = 3.0):
    """generated where keep is False, the real frame where it is True, with soft edges."""
    from PIL import Image, ImageFilter
    real = real.convert("RGB").resize(generated.size)
    m = Image.fromarray((keep * 255).astype(np.uint8)).resize(generated.size, Image.BILINEAR)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(real, generated.convert("RGB"), m)


def agnostic(img, cat: np.ndarray, grey=(128, 128, 128)):
    """The outfit region (SegFormer upper/skirt/pants/dress/belt/scarf, slightly dilated) greyed out."""
    from PIL import Image, ImageFilter
    m = Image.fromarray((np.isin(cat, OUTFIT_SF) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    return Image.composite(Image.new("RGB", img.size, grey), img.convert("RGB"), m)


def crop_part(img, cat: np.ndarray, labels: tuple[int, ...]):
    """The person's own garment (e.g. their pants) cut out on white, for the LoRA's other slot."""
    from PIL import Image
    m = np.isin(cat, labels)
    if m.sum() < 50:
        return None
    ys, xs = np.where(m)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    white = Image.new("RGB", img.size, (255, 255, 255))
    cut = Image.composite(img.convert("RGB"), white, Image.fromarray((m * 255).astype(np.uint8)))
    return cut.crop(box)


def tryon_refs(img, cat: np.ndarray, garment, category: str | None):
    """(top, bottom) reference images for the xocialize LoRA: the new garment in its slot, the
    person's own other garment in the other slot (a dress goes in both)."""
    c = str(category or "upper").lower()
    if c in ("dress", "dresses"):
        return garment, garment
    if c in ("lower", "bottoms"):
        own = crop_part(img, cat, (SF["upper"],))
        return (own if own is not None else garment), garment
    own = crop_part(img, cat, (SF["pants"], SF["skirt"]))
    return garment, (own if own is not None else garment)


TRYON_PROMPT = ("TRYON a full-body photo of a person. Replace the outfit with the top and bottom as shown in the "
                "reference images. The final image is a full body shot.")
