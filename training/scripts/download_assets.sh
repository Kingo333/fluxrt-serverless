#!/usr/bin/env bash
# Phase B: download everything training needs onto the training volume.
# Run on a CHEAP CPU Pod with the azyah-vton-train volume attached (no GPU needed):
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts/download_assets.sh | bash
# About 50 GB. Safe to re-run: finished files are skipped.
set -euo pipefail
SCRIPTS_URL="${SCRIPTS_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts}"
source <(curl -fsSL "$SCRIPTS_URL/common.sh")

command -v pip3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3-pip git >/dev/null; }
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
python3 -m pip install -q --upgrade "huggingface_hub[cli]>=0.34"
df -h "$AZ_ROOT" | tail -1

echo "== 1/5 upstream code @ $JOYAI_COMMIT"
if [ ! -d "$AZ_UPSTREAM/.git" ]; then
  git clone -q https://github.com/jd-opensource/JoyAI-Video-Edit.git "$AZ_UPSTREAM"
fi
git -C "$AZ_UPSTREAM" fetch -q origin && git -C "$AZ_UPSTREAM" checkout -q "$JOYAI_COMMIT"

echo "== 2/5 bidirectional DiT (32.5 GB) + VAE (1.5 GB)"
hf download jdopensource/JoyAI-Video-Edit --repo-type model --local-dir "$AZ_CKPT" \
  --include "dit/pretrain_editing_bid_version.pth" "vae/*"

echo "== 3/5 text encoder MiMo-VL-7B-RL-2508 (~16 GB)"
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model --local-dir "$AZ_CKPT/MiMo-VL-7B-RL-2508"

echo "== 4/5 ToyData"
hf download EasonXiao-888/JoyAI-Video-Edit_ToyData --repo-type dataset --local-dir "$AZ_DATA/toy"
# Point the manifests at the local TAR files (upstream training/README.md step 1).
python3 - "$AZ_DATA/toy" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for task in ("t2i", "i2i", "t2v", "v2v"):
    p = root / f"{task}.json"
    m = json.loads(p.read_text())
    shards = [root / Path(s).name for s in m["resolved_files"]]
    missing = [str(s) for s in shards if not s.is_file()]
    if missing:
        raise SystemExit(f"missing TAR files for {task}: {missing}")
    m["resolved_files"] = [str(s) for s in shards]
    p.write_text(json.dumps(m, indent=2) + "\n")
    print(f"manifest ok: {task} ({len(shards)} shard(s))")
PY

echo "== 5/5 verify"
fail=0
for f in "$JOYAI_BASE_MODEL" "$JOYAI_VAE/config.json" "$JOYAI_VAE/diffusion_pytorch_model.safetensors" \
         "$JOYAI_TEXT_ENCODER/config.json" "$AZ_DATA/toy/v2v.json"; do
  if [ -s "$f" ]; then echo "ok      $f"; else echo "MISSING $f"; fail=1; fi
done
size=$(stat -c %s "$JOYAI_BASE_MODEL" 2>/dev/null || echo 0)
[ "$size" -eq 32527642399 ] && echo "ok      DiT size matches Hugging Face (32527642399 bytes)" || { echo "BAD     DiT size $size (expected 32527642399)"; fail=1; }
du -sh "$AZ_CKPT" "$AZ_DATA"
[ "$fail" -eq 0 ] && echo "Done. Training assets are on the volume." || { echo "FAILED: see MISSING/BAD lines above."; exit 1; }
