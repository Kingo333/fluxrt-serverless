#!/usr/bin/env bash
# FluxRT garment LoRA (lite / backup path) on ONE GPU Pod: 1x RTX 5090 or 1x RTX PRO 6000.
# Paid: run only after the owner's "yes" with a budget.
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/run_train.sh | \
#     AZ_REF=<SHA> AZ_MANIFEST_DIR=/workspace/azyah-vton/manifests/research-train AZ_EXP=fluxlora-exp1 bash
#
# Trainer: diffusers examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py (Apache-2.0)
# at a pinned commit, plus training/patches/flux2-klein-garment.patch (second condition
# image = garment, white-padded to 384x512; caption embeddings shared).
# Data: the same r2i triples as JoyAI (garment, pseudo-source person, real target).
# Output: a diffusers LoRA (pytorch_lora_weights.safetensors). FluxRT loads it with
# use_lora + lora_weights_path. Auto-resumes from the newest checkpoint of the same AZ_EXP.
set -euo pipefail
AZ_REF="${AZ_REF:-$(curl -fsSL https://api.github.com/repos/Kingo333/fluxrt-serverless/commits/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')}"
export AZ_REF
RAW="${RAW_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training}"
echo "[azyah] scripts ref: $AZ_REF"
source <(curl -fsSL "$RAW/scripts/common.sh")
: "${AZ_MANIFEST_DIR:?set AZ_MANIFEST_DIR (output of data_tools.manifests)}"

DIFFUSERS_COMMIT="${DIFFUSERS_COMMIT:-86ef420b17363c28e36ebad6b58555e9aa8feda2}"
KLEIN_REPO="${KLEIN_REPO:-black-forest-labs/FLUX.2-klein-4B}"   # Apache-2.0; the model FluxRT serves
AZ_EXP="${AZ_EXP:-fluxlora}"
MAX_STEPS="${AZ_MAX_STEPS:-1500}"
SAMPLES="${AZ_SAMPLES:-3000}"
VIDEO_FRAMES="${AZ_VIDEO_FRAMES:-2}"   # stills per ViViD/own clip pair (research-flagged via selection.json)
BATCH="${AZ_BATCH:-4}"
RANK="${AZ_LORA_RANK:-32}"
SIZE="${AZ_SIZE:-288x512}"        # = endpoint FLUXRT_WIDTH x FLUXRT_HEIGHT
REF_SIZE="${AZ_REF_SIZE:-384x512}" # = FLUXRT_REF_WIDTH x FLUXRT_REF_HEIGHT
W="${SIZE%x*}"; H="${SIZE#*x}"
VENV="$AZ_ROOT/venv-fluxlora"
SRC="$AZ_ROOT/diffusers"
KLEIN="$AZ_CKPT/FLUX.2-klein-4B"
DATA="$AZ_DATA/fluxrt_lora/$AZ_EXP"
OUT="$AZ_OUT/fluxrt_lora/$AZ_EXP"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$AZ_REPORTS/fluxlora_${AZ_EXP}_$STAMP.log"
echo "{\"fluxrt_lora\": \"$AZ_EXP\", \"diffusers_commit\": \"$DIFFUSERS_COMMIT\", \"base\": \"$KLEIN_REPO\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$AZ_REPORTS/code_refs.jsonl"

echo "== 1. env (once)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
fi
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/huggingface/diffusers.git "$SRC"
fi
git -C "$SRC" fetch -q origin "$DIFFUSERS_COMMIT" || true
git -C "$SRC" checkout -q -f "$DIFFUSERS_COMMIT"
curl -fsSL "$RAW/patches/flux2-klein-garment.patch" -o "$AZ_ROOT/flux2-klein-garment.patch"
git -C "$SRC" apply "$AZ_ROOT/flux2-klein-garment.patch"
"$VENV/bin/pip" install -q -e "$SRC" accelerate transformers peft datasets bitsandbytes tensorboard sentencepiece pyarrow pillow huggingface_hub
"$VENV/bin/python" -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__)"

echo "== 2. base model (once, ~16 GB on the volume)"
if [ ! -f "$KLEIN/.downloaded" ]; then
  "$VENV/bin/huggingface-cli" download "$KLEIN_REPO" --local-dir "$KLEIN"
  touch "$KLEIN/.downloaded"
fi

echo "== 3. dataset ($SAMPLES triples at $SIZE)"
if [ ! -f "$DATA/dataset_summary.json" ]; then
  mkdir -p "$AZ_ROOT/azyah-tools"
  for f in data_tools/__init__.py data_tools/fluxrt_prompt.py fluxrt_lora/__init__.py fluxrt_lora/make_dataset.py; do
    mkdir -p "$AZ_ROOT/azyah-tools/$(dirname "$f")"
    curl -fsSL "$RAW/$f" -o "$AZ_ROOT/azyah-tools/$f"
  done
  command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
  "$VENV/bin/python" "$AZ_ROOT/azyah-tools/fluxrt_lora/make_dataset.py" --manifest-dir "$AZ_MANIFEST_DIR" --out "$DATA" \
    --max "$SAMPLES" --size "$SIZE" --video-frames "$VIDEO_FRAMES"
fi
cat "$DATA/dataset_summary.json"
[ -f "$DATA/selection.json" ] && { echo "== data selection (provenance)"; cat "$DATA/selection.json"; }

echo "== 4. train (resumes from the latest checkpoint)"
mkdir -p "$OUT"
cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "${LOG%.log}_gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT
cd "$SRC/examples/dreambooth"
"$VENV/bin/accelerate" launch --num_processes 1 --mixed_precision bf16 train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path="$KLEIN" \
  --dataset_name="$DATA/data" \
  --image_column=target --cond_image_column=person --cond_image_column_2=garment --caption_column=caption \
  --cond_2_size="$REF_SIZE" \
  --use_aspect_ratio_buckets --aspect_ratio_buckets="$H,$W" --resolution="$H" \
  --max_sequence_length=512 \
  --output_dir="$OUT" \
  --rank="$RANK" --lora_alpha="$RANK" \
  --train_batch_size="$BATCH" --gradient_accumulation_steps=1 \
  --gradient_checkpointing --cache_latents \
  --mixed_precision=bf16 --guidance_scale=1 \
  --optimizer=adamW --use_8bit_adam --learning_rate="${AZ_LR:-1e-4}" \
  --lr_scheduler=constant_with_warmup --lr_warmup_steps=50 \
  --max_train_steps="$MAX_STEPS" \
  --checkpointing_steps="${AZ_CKPT_INTERVAL:-100}" --checkpoints_total_limit=2 \
  --resume_from_checkpoint=latest \
  --skip_final_inference --report_to=tensorboard \
  --seed=0 2>&1 | tee "$LOG"
echo "== done"; ls -la "$OUT" | tail -8
echo "LoRA: $OUT/pytorch_lora_weights.safetensors (+ selection.json for the model card)"
