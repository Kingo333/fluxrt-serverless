"""Write a garment description for staged pairs that have none (review R2), in the
same style as the app's Gemini wardrobe analysis (concrete visible features only),
so pack.py can add the detailed caption the live app sends.

  GEMINI_API_KEY=... python3 -m data_tools.caption_garments STAGING_DIR [--limit N] [--model gemini-2.5-flash]
  python3 -m data_tools.caption_garments STAGING_DIR --backend mimo \
      --model /workspace/azyah-vton/checkpoints/MiMo-VL-7B-RL-2508      # local, on the training GPU Pod

Uses only the garment photo (ref.*), never the person. Gemini is paid API usage; mimo runs
MiMo-VL-7B-RL-2508 (JoyAI's own text encoder, already on the training volume) locally on the
GPU Pod. Either way: owner's approval and a budget first. Results go to meta.json as
garment_description (+ garment_caption_model); pack.py turns it into caption_detailed, the
app's buildTryOnInstruction() full form ("The reference garment: <details>. Match its exact ...").

Step 3a (owner's step 1 review: trims, prints, texture, hems and length were simplified): the
mimo prompt asks for exactly those details, in <= 25 words.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Protocol

from .staging import clean_product_text, find, iter_pairs

PROMPT = (
    "Describe this clothing item for a virtual try-on system in one or two sentences (max 45 words). "
    "Name only features you can clearly see: garment type, colours, pattern or print, any logo or text "
    "and where it is, neckline, sleeves, fabric look. No guessing, no styling advice, no brand claims."
)


DETAIL_PROMPT = (
    "Describe this clothing item for a virtual try-on system in at most 25 words, as one plain phrase. "
    "Name only what you clearly see: garment type, colours, print or pattern (motif, size, placement), "
    "neckline or collar (bow, tie, ruffle), trims (ruffles, pleats, buttons), sleeves, length or hem, "
    "fabric texture (knit, rib, lace, denim). No guessing, no styling advice, no brand claims. /no_think"
)
MAX_WORDS = 25


def tidy_description(text: str, max_words: int = MAX_WORDS) -> str:
    """Model output -> one clean phrase: drop any <think> block, quotes and labels, cap words."""
    import re
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = re.sub(r"^\s*(description|garment)\s*:\s*", "", text.strip(), flags=re.I)
    text = clean_product_text(text.strip().strip('"').strip("'"))
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:")
    return text


class Captioner(Protocol):
    name: str

    def describe(self, image_bytes: bytes, mime: str) -> str: ...


class GeminiCaptioner:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.name = model
        self.key = os.environ["GEMINI_API_KEY"]

    def describe(self, image_bytes: bytes, mime: str) -> str:
        import urllib.request
        body = {"contents": [{"parts": [{"text": PROMPT},
                                        {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120}}
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.name}:generateContent",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        return data["candidates"][0]["content"]["parts"][0]["text"]


class MiMoVLCaptioner:
    """Local MiMo-VL-7B-RL-2508 (Qwen2.5-VL layout) through transformers, greedy, on the GPU."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:  # older transformers
            from transformers import Qwen2_5_VLForConditionalGeneration as Auto
        self.name = f"mimo-vl:{Path(model_dir).name}"
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Auto.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device = device

    def describe(self, image_bytes: bytes, mime: str) -> str:
        import io
        import torch
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((768, 768))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DETAIL_PROMPT}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return tidy_description(reply)


def has_input(d: Path, meta: dict) -> bool:
    """A pair that will be trained on: it has a P8 source.* or an ok live_source.*."""
    return find(d, "source") is not None or bool((meta.get("live_source") or {}).get("ok") and find(d, "live_source"))


def process(staging_dir: str, captioner: Captioner, limit: int | None = None, only_with_source: bool = False) -> dict:
    stats = {"described": 0, "skipped": 0, "failed": 0}
    for d, meta in iter_pairs(Path(staging_dir)):
        if meta.get("garment_description") or (only_with_source and not has_input(d, meta)):
            stats["skipped"] += 1
            continue
        if limit is not None and stats["described"] + stats["failed"] >= limit:
            break
        ref = find(d, "ref")
        mime = "image/png" if ref.suffix == ".png" else "image/jpeg"
        try:
            text = clean_product_text(captioner.describe(ref.read_bytes(), mime))
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            meta["garment_caption_error"] = type(e).__name__
        else:
            if text:
                meta["garment_description"] = text
                meta["garment_caption_model"] = captioner.name
                stats["described"] += 1
            else:
                stats["failed"] += 1
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging_dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", default="gemini", choices=["gemini", "mimo"])
    ap.add_argument("--only-with-source", action="store_true", help="only pairs that have a source/live_source input")
    ap.add_argument("--model", help="gemini model name, or the MiMo-VL folder for --backend mimo")
    a = ap.parse_args()
    if a.backend == "mimo":
        cap = MiMoVLCaptioner(a.model or "/workspace/azyah-vton/checkpoints/MiMo-VL-7B-RL-2508")
    else:
        cap = GeminiCaptioner(a.model or "gemini-2.5-flash")
    print(json.dumps(process(a.staging_dir, cap, a.limit, a.only_with_source), indent=2))


if __name__ == "__main__":
    main()
