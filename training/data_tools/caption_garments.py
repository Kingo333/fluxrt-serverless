"""Write a garment description for staged pairs that have none (review R2), in the
same style as the app's Gemini wardrobe analysis (concrete visible features only),
so pack.py can add the detailed caption the live app sends.

  GEMINI_API_KEY=... python3 -m data_tools.caption_garments STAGING_DIR [--limit N] [--model gemini-2.5-flash]

Uses only the garment photo (ref.*), never the person. Paid API usage: run only
with the owner's approval and a budget. Results go to meta.json as
garment_description (+ garment_caption_model).
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


def process(staging_dir: str, captioner: Captioner, limit: int | None = None) -> dict:
    stats = {"described": 0, "skipped": 0, "failed": 0}
    for d, meta in iter_pairs(Path(staging_dir)):
        if meta.get("garment_description"):
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
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()
    print(json.dumps(process(a.staging_dir, GeminiCaptioner(a.model), a.limit), indent=2))


if __name__ == "__main__":
    main()
