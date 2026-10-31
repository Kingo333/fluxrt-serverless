"""Garment masks for staged pairs that lack one (own clips, ViViD): Grounding DINO
finds the garment box, SAM 2.1 cuts the mask. Both Apache-2.0, both from
`transformers` (no extra packages, no Ultralytics/AGPL). GPU recommended; works on CPU slowly.

  python3 -m data_tools.masks STAGING_DIR [--only own] [--device cuda]

Writes mask.png (white = garment area) from the middle frame of target.mp4 or from
target.png. Used offline only: to filter pseudo-sources and to score the garment
region in evaluation. Never a model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .staging import find, iter_pairs

DETECT_TEXT = {"upper": "a shirt. a top. a jacket.", "lower": "trousers. a skirt. shorts.", "dress": "a dress."}
DINO = "IDEA-Research/grounding-dino-base"
SAM = "facebook/sam2.1-hiera-large"


def pick_box(boxes, scores, min_score: float = 0.3):
    """Highest-scoring box above the threshold, or None. boxes: [[x0,y0,x1,y1], ...]."""
    best = None
    for b, s in zip(boxes, scores):
        if s >= min_score and (best is None or s > best[1]):
            best = (list(map(float, b)), float(s))
    return best


def _middle_frame(path: Path):
    from PIL import Image
    if path.suffix != ".mp4":
        return Image.open(path).convert("RGB")
    import av
    with av.open(str(path)) as c:
        frames = [f.to_image() for f in c.decode(c.streams.video[0])]
    return frames[len(frames) // 2].convert("RGB")


class Masker:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.torch, self.device = torch, device
        self.dino_proc = AutoProcessor.from_pretrained(DINO)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(SAM)
        self.sam = Sam2Model.from_pretrained(SAM).to(device).eval()

    def mask(self, image, category: str):
        torch = self.torch
        with torch.no_grad():
            inp = self.dino_proc(images=image, text=DETECT_TEXT[category], return_tensors="pt").to(self.device)
            out = self.dino(**inp)
            det = self.dino_proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
            best = pick_box(det["boxes"].tolist(), det["scores"].tolist())
            if best is None:
                return None, 0.0
            sin = self.sam_proc(images=image, input_boxes=[[best[0]]], return_tensors="pt").to(self.device)
            sout = self.sam(**sin, multimask_output=False)
            m = self.sam_proc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0][0, 0]
        from PIL import Image
        return Image.fromarray((m.numpy() > 0.5).astype("uint8") * 255), best[1]


def process(staging_dir: str, masker, only: str | None = None) -> dict:
    stats = {"masked": 0, "no_garment_found": 0, "skipped": 0}
    for d, meta in iter_pairs(Path(staging_dir)):
        if find(d, "mask") or (only and meta["source"] != only):
            stats["skipped"] += 1
            continue
        m, score = masker.mask(_middle_frame(find(d, "target")), meta["category"])
        if m is None:
            stats["no_garment_found"] += 1
            meta["mask"] = {"ok": False}
        else:
            m.save(d / "mask.png")
            meta["mask"] = {"ok": True, "detector_score": round(score, 3), "model": f"{DINO} + {SAM}"}
            stats["masked"] += 1
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging_dir")
    ap.add_argument("--only")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    print(json.dumps(process(a.staging_dir, Masker(a.device), a.only), indent=2))


if __name__ == "__main__":
    main()
