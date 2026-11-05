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
UPPER_LIMIT="${AZ_LIVE_UPPER:-500}"      # live-way inputs per Glance set (~3 s each on a PRO 6000)
LOWER_LIMIT="${AZ_LIVE_LOWER:-500}"
DRESS_LIMIT="${AZ_VIVID_DRESS_LIMIT:-400}"
# Owner decision 2026-11-05: ViViD tops and bottoms too (real frames as targets, inputs made the
# live way). Research-only like the dresses; AZ_VIVID_STILLS=0 keeps only the dresses.
VIVID_STILLS="${AZ_VIVID_STILLS:-1}"
VUPPER_LIMIT="${AZ_VIVID_UPPER_LIMIT:-400}"
VLOWER_LIMIT="${AZ_VIVID_LOWER_LIMIT:-343}"
# Real-motion clips (owner 2026-11-05: accuracy while the person moves, turns, shows side/back):
# ViViD train clips get an input made the live way (the 0811 clone, a fixed garment photo of
# another item, 97 frames), kept only if steady (live_source.clip_check); target = the real clip.
VIVID_CLIPS="${AZ_VIVID_CLIPS:-1}"
CLIPS_PER_CAT="${AZ_VIVID_CLIPS_PER_CAT:-150}"
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
    n_dress=$(find "$S/vivid_dress_stills" -mindepth 2 -maxdepth 2 -name meta.json | wc -l)
    echo "vivid dress stills: $n_dress"
    [ "$n_dress" -gt 0 ] || { echo "NO_DRESS_STILLS: see the 'seen' counts above (split/category/target/ref)"; exit 1; }
  fi
  echo "CPU_PART_DONE"
  exit 0
fi

[ "$AZ_PART" = "gpu" ] || { echo "AZ_PART must be cpu or gpu"; exit 2; }
DIT="$(readlink -f "$(find "$C" -name joyai_video_edit_dit_0811.pth | head -1)")"
case "$DIT" in "$AZ_ROOT"/*) ;; *) echo "REFUSING: 0811 clone not on the training volume ($DIT)"; exit 1 ;; esac
# AZ_ENGINE=trainpipe (recommended since 2026-11-05): JoyAI's training streaming pipeline with the
# training venv on the volume, on the plain RunPod PyTorch template (starts in 1-2 min; the endpoint
# image hung three Pods at "Initializing"). Same live settings as step 1 (generate_vivid --live).
# AZ_ENGINE=deploy: the endpoint image's JoyAI runtime (/opt/joyai-venv), as P8 ran.
ENGINE="${AZ_ENGINE:-deploy}"
if [ "$ENGINE" = "trainpipe" ]; then
  export SCRIPTS_URL="https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/scripts"
  curl -fsSL "$SCRIPTS_URL/setup_training_pod.sh" | bash          # training venv + upstream patches (SDPA)
  PY="$AZ_ROOT/venv/bin/python"
  UP="$AZ_ROOT/JoyAI-Video-Edit/training"
  "$PY" "$AZ_ROOT/azyah-code/training/scripts/make_fashion_config.py" "$UP"
  export JOYAI_BASE_MODEL="$DIT" JOYAI_VAE="$C/vae" JOYAI_TEXT_ENCODER="$C/MiMo-VL-7B-RL-2508"
  # The fashion config refuses to load without a manifest folder; the editor reads no training data,
  # so any existing manifest will do (the same one step 1's generate_vivid used).
  export AZ_MANIFEST_DIR="${AZ_MANIFEST_DIR:-$AZ_ROOT/manifests/research-train}"
  export PYTHONPATH="$UP:$AZ_ROOT/azyah-code/training" PATH="$AZ_ROOT/venv/bin:$PATH"
  LS=("$AZ_ROOT/venv/bin/torchrun" --nproc_per_node 1 -m data_tools.live_source --dit "$DIT" --vae "$C/vae"
      --text-encoder "$C/MiMo-VL-7B-RL-2508" --engine trainpipe --config "$UP/configs/azyah_fashion_stage1.py")
else
  PY=/opt/joyai-venv/bin/python
  [ -x "$PY" ] || { echo "no /opt/joyai-venv: start this Pod from the endpoint image, or use AZ_ENGINE=trainpipe"; exit 1; }
  export TORCHDYNAMO_DISABLE=1 JOYOMNI_LOW_VRAM=1 JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1 JOYOMNI_FP8_IMG=1 \
    JOYOMNI_FP8_TXT=1 JOYOMNI_CUDA_GRAPH=1 JOYOMNI_WARMUP_ORIENTATION=portrait
  export TORCHINDUCTOR_CACHE_DIR="$AZ_ROOT/cache_p8/inductor" TRITON_CACHE_DIR="$AZ_ROOT/cache_p8/triton" \
    CUDA_CACHE_PATH="$AZ_ROOT/cache_p8/nv" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export JOYAI_DEPLOY_DIR=/opt/JoyAI-Video-Edit/deploy PYTHONPATH="/opt/JoyAI-Video-Edit/deploy:$PWD"
  LS=("$PY" -m data_tools.live_source --dit "$DIT" --vae "$C/vae" --text-encoder "$C/MiMo-VL-7B-RL-2508")
fi
cd "$AZ_ROOT/azyah-code/training"
if [ "$VIVID_STILLS" = "1" ]; then
  echo "== 0. ViViD top and bottom stills (one steady real frame per train clip; research-only)"
  "$PY" -c "import av" 2>/dev/null || "$PY" -m pip install -q av
  [ -d "$S/vivid_upper_stills" ] || "$PY" -m data_tools.vivid_stills "$S/vivid" "$S/vivid_upper_stills" --category upper --limit "$VUPPER_LIMIT"
  [ -d "$S/vivid_lower_stills" ] || "$PY" -m data_tools.vivid_stills "$S/vivid" "$S/vivid_lower_stills" --category lower --limit "$VLOWER_LIMIT"
fi
DIRS=("$S/glance_upper" "$S/glance_lower")
VIVID_SETS=()
[ "$VIVID_DRESSES" = "1" ] && [ -d "$S/vivid_dress_stills" ] && VIVID_SETS+=("dress")
[ "$VIVID_STILLS" = "1" ] && [ -d "$S/vivid_upper_stills" ] && VIVID_SETS+=("upper")
[ "$VIVID_STILLS" = "1" ] && [ -d "$S/vivid_lower_stills" ] && VIVID_SETS+=("lower")
for v in "${VIVID_SETS[@]}"; do DIRS+=("$S/vivid_${v}_stills"); done
LS+=(--remove-frac "$REMOVE_FRAC")

echo "== 1. live-way inputs (fixed garment photo of another item, ${REMOVE_FRAC} remove examples)"
# One call per set so each has its own limit; the reference pool is every set (same category only).
timeout -s INT 75m "${LS[@]}" "$S/glance_upper" --limit "$UPPER_LIMIT" || echo "UPPER_EXIT=$?"
timeout -s INT 75m "${LS[@]}" "$S/glance_lower" --limit "$LOWER_LIMIT" || echo "LOWER_EXIT=$?"
for v in "${VIVID_SETS[@]}"; do
  case "$v" in dress) lim="$DRESS_LIMIT" ;; upper) lim="$VUPPER_LIMIT" ;; *) lim="$VLOWER_LIMIT" ;; esac
  timeout -s INT 45m "${LS[@]}" "$S/vivid_${v}_stills" --limit "$lim" || echo "VIVID_${v}_EXIT=$?"
done

if [ "$VIVID_CLIPS" = "1" ]; then
  echo "== 1b. real-motion clips: ViViD train clips, inputs made the live way (${CLIPS_PER_CAT} per category, 97 frames)"
  for c in dress upper lower; do
    LSV=("${LS[@]}"); LSV[${#LSV[@]}-1]=0   # last arg of LS is the remove fraction: none for clips
    timeout -s INT 40m "${LSV[@]}" "$S/vivid" --videos --categories "$c" --max-frames 97 \
      --limit "$CLIPS_PER_CAT" || echo "CLIPS_${c}_EXIT=$?"
  done
fi

echo "== 2. detail captions (MiMo-VL-7B-RL-2508, local; only pairs that have an input)"
CAP_LIMIT="${AZ_CAPTION_LIMIT:-1500}"   # per set, ~1 s each; half the training captions use them anyway
CAP_DIRS=("${DIRS[@]}"); [ "$VIVID_CLIPS" = "1" ] && CAP_DIRS+=("$S/vivid")
for d in "${CAP_DIRS[@]}"; do
  if [ "$d" = "$S/vivid" ]; then
    timeout -s INT 20m "$PY" -m data_tools.caption_garments "$d" --backend mimo --model "$C/MiMo-VL-7B-RL-2508" \
      --only-live-source --limit 500 || echo "CAPTION_EXIT=$? ($d)"
    continue
  fi
  timeout -s INT 30m "$PY" -m data_tools.caption_garments "$d" --backend mimo --model "$C/MiMo-VL-7B-RL-2508" \
    --only-with-source --limit "$CAP_LIMIT" || echo "CAPTION_EXIT=$? ($d)"
done

echo "== 3. pack (live_source preferred) + manifests"
"$PY" -m data_tools.pack "$S/glance_upper" "$D/shards" --name glance_upper_live --prefer-live-source
"$PY" -m data_tools.pack "$S/glance_lower" "$D/shards" --name glance_lower_live --prefer-live-source
"$PY" -m data_tools.manifests "$D/shards/glance_upper_live.catalog.json" "$D/shards/glance_lower_live.catalog.json" \
  --out "$M/joyai0811-commercial-train" --mode commercial --split train
if [ "${#VIVID_SETS[@]}" -gt 0 ]; then
  CATS=("$D/shards/glance_upper_live.catalog.json" "$D/shards/glance_lower_live.catalog.json")
  for v in "${VIVID_SETS[@]}"; do
    "$PY" -m data_tools.pack "$S/vivid_${v}_stills" "$D/shards" --name "vivid_${v}_live" --prefer-live-source
    CATS+=("$D/shards/vivid_${v}_live.catalog.json")
  done
  if [ "$VIVID_CLIPS" = "1" ]; then
    "$PY" -m data_tools.pack "$S/vivid" "$D/shards" --name vivid_clips_live --only-live-source --split train
    CATS+=("$D/shards/vivid_clips_live.catalog.json")
  fi
  "$PY" -m data_tools.manifests "${CATS[@]}" --out "$M/joyai0811-research-train" --mode research --split train
fi
for m in "$M/joyai0811-commercial-train" "$M/joyai0811-research-train"; do
  [ -f "$m/selection.json" ] && { echo "== $m"; cat "$m/selection.json"; }
done
echo "== counts"
"$PY" - "${CAP_DIRS[@]}" <<'PYEOF'
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
