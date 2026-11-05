#!/usr/bin/env bash
# Step 3b: data for the 0811 LoRA (step 3a changes). Two parts, both resumable, volume azyah-vton-train
# at /workspace. Paid: owner's "yes" first. Never touches the live endpoint or the live azyah-joyai volume.
#
#   AZ_PART=cpu  (CPU Pod, ~$0.10): Glance lowerwear download + staging (glance_lower), and with
#                AZ_VIVID_DRESSES=1 (default, owner's Option A) one steady frame per ViViD train dress
#                clip (vivid_dress_stills, research-only).
#   AZ_PART=gpu  (1x RTX PRO 6000, Pod template = the endpoint image, as P8 ran; /opt/joyai-venv):
#                live-way inputs (data_tools.live_source: the 0811 CLONE with a fixed garment photo,
#                25% "remove" examples), MiMo-VL detail captions, packing and two manifests:
#                  manifests/joyai0811-commercial-train  (Glance upper + lower: commercial-clean)
#                  manifests/joyai0811-research-train    (+ ViViD dress stills: research-only)
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/joyai_0811_lora/build_data.sh | \
#     AZ_REF=<SHA> AZ_PART=cpu bash
set -euo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
AZ_PART="${AZ_PART:?set AZ_PART=cpu or AZ_PART=gpu}"
export AZ_ROOT="${AZ_ROOT:-/workspace/azyah-vton}"
# Plain Hugging Face downloads: the PyTorch template sets HF_HUB_ENABLE_HF_TRANSFER=1 without the
# hf_transfer package (3b CPU run failed on it); same settings as scripts/common.sh.
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
C="$AZ_ROOT/checkpoints"; S="$AZ_ROOT/data/staging"; D="$AZ_ROOT/data"; M="$AZ_ROOT/manifests"
VIVID_DRESSES="${AZ_VIVID_DRESSES:-1}"
UPPER_LIMIT="${AZ_LIVE_UPPER:-1200}"     # live-way inputs per Glance set (~3 s each on a PRO 6000)
LOWER_LIMIT="${AZ_LIVE_LOWER:-1200}"
DRESS_LIMIT="${AZ_VIVID_DRESS_LIMIT:-400}"
REMOVE_FRAC="${AZ_REMOVE_FRAC:-0.25}"
GLANCE_REV=7ae72513413932dbfdb15d02c924f3be9739b0c4
mkdir -p "$AZ_ROOT/reports"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF"
cd "$AZ_ROOT/azyah-code/training"
echo "{\"joyai0811_data\": \"$AZ_PART\", \"azyah_scripts_ref\": \"$AZ_REF\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$AZ_ROOT/reports/code_refs.jsonl"

if [ "$AZ_PART" = "cpu" ]; then
  PY="$AZ_ROOT/venv/bin/python"; HF="$AZ_ROOT/venv/bin/hf"
  echo "== Glance lowerwear (CC-BY-4.0, commercial-ok), pinned revision"
  for cfg in mens-lowerwear womens-lowerwear; do
    "$HF" download glanceai/VTON-Synthetic-Pairs --repo-type dataset --revision "$GLANCE_REV" \
      --local-dir "$D/raw/glance" --include "$cfg/*"
  done
  "$PY" -m data_tools.convert_glance "$D/raw/glance" "$S/glance_lower" --configs mens-lowerwear womens-lowerwear --per-config 2500
  ls "$S/glance_lower" | wc -l
  if [ "$VIVID_DRESSES" = "1" ]; then
    echo "== ViViD dress stills (one steady real frame per train clip; research-only)"
    "$PY" -c "import av" 2>/dev/null || "$AZ_ROOT/venv/bin/pip" install -q av
    "$PY" -m data_tools.vivid_stills "$S/vivid" "$S/vivid_dress_stills" --category dress --limit "$DRESS_LIMIT"
  fi
  echo "CPU_PART_DONE"
  exit 0
fi

[ "$AZ_PART" = "gpu" ] || { echo "AZ_PART must be cpu or gpu"; exit 2; }
PY=/opt/joyai-venv/bin/python
[ -x "$PY" ] || { echo "no /opt/joyai-venv: start this Pod from the endpoint image (as P8 did)"; exit 1; }
DIT="$(readlink -f "$(find "$C" -name joyai_video_edit_dit_0811.pth | head -1)")"
case "$DIT" in "$AZ_ROOT"/*) ;; *) echo "REFUSING: 0811 clone not on the training volume ($DIT)"; exit 1 ;; esac
export TORCHDYNAMO_DISABLE=1 JOYOMNI_LOW_VRAM=1 JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1 JOYOMNI_FP8_IMG=1 \
  JOYOMNI_FP8_TXT=1 JOYOMNI_CUDA_GRAPH=1 JOYOMNI_WARMUP_ORIENTATION=portrait
export TORCHINDUCTOR_CACHE_DIR="$AZ_ROOT/cache_p8/inductor" TRITON_CACHE_DIR="$AZ_ROOT/cache_p8/triton" \
  CUDA_CACHE_PATH="$AZ_ROOT/cache_p8/nv" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JOYAI_DEPLOY_DIR=/opt/JoyAI-Video-Edit/deploy PYTHONPATH="/opt/JoyAI-Video-Edit/deploy:$PWD"
DIRS=("$S/glance_upper" "$S/glance_lower")
[ "$VIVID_DRESSES" = "1" ] && [ -d "$S/vivid_dress_stills" ] && DIRS+=("$S/vivid_dress_stills")
LS=(-m data_tools.live_source --dit "$DIT" --vae "$C/vae" --text-encoder "$C/MiMo-VL-7B-RL-2508" --remove-frac "$REMOVE_FRAC")

echo "== 1. live-way inputs (fixed garment photo of another item, ${REMOVE_FRAC} remove examples)"
# One call per set so each has its own limit; the reference pool is every set (same category only).
timeout -s INT 75m "$PY" "${LS[@]}" "$S/glance_upper" --limit "$UPPER_LIMIT" || echo "UPPER_EXIT=$?"
timeout -s INT 75m "$PY" "${LS[@]}" "$S/glance_lower" --limit "$LOWER_LIMIT" || echo "LOWER_EXIT=$?"
if [ "$VIVID_DRESSES" = "1" ] && [ -d "$S/vivid_dress_stills" ]; then
  timeout -s INT 30m "$PY" "${LS[@]}" "$S/vivid_dress_stills" --limit "$DRESS_LIMIT" || echo "DRESS_EXIT=$?"
fi

echo "== 2. detail captions (MiMo-VL-7B-RL-2508, local; only pairs that have an input)"
CAP_LIMIT="${AZ_CAPTION_LIMIT:-1500}"   # per set, ~1 s each; half the training captions use them anyway
for d in "${DIRS[@]}"; do
  timeout -s INT 30m "$PY" -m data_tools.caption_garments "$d" --backend mimo --model "$C/MiMo-VL-7B-RL-2508" \
    --only-with-source --limit "$CAP_LIMIT" || echo "CAPTION_EXIT=$? ($d)"
done

echo "== 3. pack (live_source preferred) + manifests"
"$PY" -m data_tools.pack "$S/glance_upper" "$D/shards" --name glance_upper_live --prefer-live-source
"$PY" -m data_tools.pack "$S/glance_lower" "$D/shards" --name glance_lower_live --prefer-live-source
"$PY" -m data_tools.manifests "$D/shards/glance_upper_live.catalog.json" "$D/shards/glance_lower_live.catalog.json" \
  --out "$M/joyai0811-commercial-train" --mode commercial --split train
if [ "$VIVID_DRESSES" = "1" ] && [ -d "$S/vivid_dress_stills" ]; then
  "$PY" -m data_tools.pack "$S/vivid_dress_stills" "$D/shards" --name vivid_dress_live --prefer-live-source
  "$PY" -m data_tools.manifests "$D/shards/glance_upper_live.catalog.json" "$D/shards/glance_lower_live.catalog.json" \
    "$D/shards/vivid_dress_live.catalog.json" --out "$M/joyai0811-research-train" --mode research --split train
fi
for m in "$M/joyai0811-commercial-train" "$M/joyai0811-research-train"; do
  [ -f "$m/selection.json" ] && { echo "== $m"; cat "$m/selection.json"; }
done
echo "== counts"
"$PY" - "${DIRS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
for d in sys.argv[1:]:
    n = ok = extra = cap = 0
    for m in Path(d).glob("*/meta.json"):
        meta = json.loads(m.read_text())
        ls = meta.get("live_source") or {}
        n += bool(ls); ok += bool(ls.get("ok")); extra += bool(ls.get("ok") and ls.get("extra_item"))
        cap += bool(meta.get("garment_description"))
    print(f"{d}: live inputs tried {n}, ok {ok}, remove examples {extra}, captioned {cap}")
PYEOF
echo "GPU_PART_DONE"
