#!/usr/bin/env bash
# FluxRT LoRA "fxreal4" (owner, 2026-11-08). fxreal3's recipe, trained fresh from the base model, plus the
# fix for what the owner saw on item 5 (a short feather top over a long top: fxreal3 drew TWO feather bands,
# keeping the old top's length):
#   - longer-to-shorter examples: for 25% of re-dressed tops the INPUT garment is drawn longer (to the
#     upper thighs) and pasted back over a band below the real hem; the answer keeps the real, shorter top,
#     so the model learns that leftover length turns back into trousers, skirt or skin
#   - the edge/texture (detail) term turned down from 0.1 to 0.03 (it may have encouraged the extra band)
#   - everything else as fxreal3: same-clip pairs, light jitter, face 2x, ~20% no-change, garment 2x,
#     real answers, paste-back
# Then the owner's same 5 clips + 10 VITON-HD pairs, the gate, the stability table, an ITEM-5-ONLY blind video
# (judged first) and the full 5-item blind video, each with its key in a separate folder.
# Needs a GPU with >= 30 GB (RTX PRO 6000 or RTX 5090) and the azyah-vton-train volume at /workspace.
# Never touches the live endpoint or the azyah-joyai volume. Approved cost ~$7-10.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx4.sh -o /workspace/fx4.sh
#   AZ_REF=<SHA> AZ_NO_STOP=1 nohup bash /workspace/fx4.sh > /workspace/fx4.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
EXP="${AZ_EXP:-fxreal4}"
GUARD=/workspace/fx4_started
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
[ -e "$GUARD" ] && [ "${AZ_FORCE:-0}" != "1" ] && { echo "GUARD: $GUARD exists (already ran); AZ_FORCE=1 to run again"; exit 0; }
touch "$GUARD"
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E="$AZ_ROOT/eval/fx2"; E3="$AZ_ROOT/eval/fx4"; R3="$E3/results"
V="$AZ_ROOT/venv-fluxlora/bin"; PY="$V/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
mkdir -p "$E3" "$R3" "$AZ_ROOT/reports"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }

echo "== 0. GPU and space"
nvidia-smi --query-gpu=name,memory.total --format=csv
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
[ "${VRAM:-0}" -ge 30000 ] || { echo "GPU_TOO_SMALL: ${VRAM} MiB; training needs >= 30 GB (RTX PRO 6000 or RTX 5090)"; exit 1; }
df -h /workspace | tail -1
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 30 ] || { echo "LOW_SPACE: ${FREE_GB} GB free on the volume"; exit 1; }

echo "== 1. code at $AZ_REF"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -f "$E/cases5/clip_list.json" ] || { echo "NO_CASES5: $E/cases5/clip_list.json (the owner's verified 5 items) is missing"; exit 1; }
[ -f "$D/raw/viton_hd/.downloaded" ] || { echo "NO_VITON: $D/raw/viton_hd not downloaded (fx2 does that)"; exit 1; }

echo "== 2. MediaPipe venv (+ pyarrow for the face weights)"
if [ ! -x "$MP" ] || ! "$MP" -c "import mediapipe, av" 2>/dev/null; then
  python3 -m venv "$AZ_ROOT/venv-mp" && "$AZ_ROOT/venv-mp/bin/pip" install -q mediapipe==0.10.21 av pillow numpy || true
fi
"$MP" -c "import pyarrow" 2>/dev/null || "$AZ_ROOT/venv-mp/bin/pip" install -q pyarrow
"$MP" -c "import mediapipe, av, pyarrow" 2>/dev/null || { echo "NO_MEDIAPIPE"; exit 1; }

echo "== 3. same-clip frame pairs from ViViD train videos ($(mins))"
( cd "$T" && "$MP" -m data_tools.vivid_pairs "$S/vivid" "$S/vivid_pairs" --n "${AZ_PAIRS:-300}" --model "$SEG" ) || echo "PAIRS_EXIT=$?"

DATA="$D/fluxrt_lora/$EXP"
if [ ! -f "$DATA/dataset_summary.json" ]; then
  echo "== 4. build the dataset: as fxreal3 + 25% longer-to-shorter tops ($(mins))"
  timeout -s INT 120m "$PY" "$T/fluxrt_lora/build_real.py" --out "$DATA" --viton-dir "$D/raw/viton_hd" \
    --klein "$C/FLUX.2-klein-4B" --vivid-stills "$S/vivid_dress_stills" "$S/vivid_upper_stills" "$S/vivid_lower_stills" \
    --vivid-pairs "$S/vivid_pairs" --test-out "$E3/unused_test_pairs" --test-items 0 \
    --viton-n "${AZ_VITON_N:-2000}" --holdout 300 --nochange-frac "${AZ_NOCHANGE:-0.33}" --photo-aug 0.3 \
    --longer-frac "${AZ_LONGER:-0.25}" --longer-ext 0.2
  echo "BUILD_EXIT=$?"
fi
cat "$DATA/dataset_summary.json" || { echo NO_DATASET; exit 1; }

echo "== 4b. face counts 2x: add the face to every sample's weight mask ($(mins))"
( cd "$T" && "$MP" -m data_tools.face_weight "$DATA" --model "$SEG" ) || { echo "FACE_WEIGHT_FAILED"; exit 1; }

echo "== 5. train fxreal4 on FLUX.2-klein-base-4B: 2,000 steps, saves every 250 ($(mins))"
OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"
if [ ! -f "$OUT/pytorch_lora_weights.safetensors" ]; then
  timeout -s INT 180m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=288x512 \
    AZ_MAX_STEPS=${AZ_MAX_STEPS:-2000} AZ_BATCH=4 AZ_LORA_RANK=32 AZ_LR=${AZ_LR:-1e-4} \
    AZ_CKPT_INTERVAL=250 AZ_CKPT_LIMIT=8 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=2.0 \
    AZ_PAIR_COLUMN=pair AZ_PAIR_WEIGHT=${AZ_PAIR_WEIGHT:-1.0} AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.03} bash"
  echo "TRAIN_EXIT=$?"
fi
ls -la "$OUT" | tail -12
[ -f "$OUT/pytorch_lora_weights.safetensors" ] || { echo LORA_MISSING; exit 1; }
cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
grep -oE "(loss|detail|pair)=[0-9.e+-]+" "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | tail -6
echo "train done after $(mins)"

echo "== 6. renders: base FluxRT and fxreal4 (final + step 1500), distilled klein-4B, 2 steps, 288x512"
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --ref-size 288x512 --steps 2 --prompt-style compact"
render() {  # name [lora-file]
  local n="$1" extra=""
  [ -n "${2:-}" ] && extra="--lora $2 --lora-scale 1.0"
  ( cd "$T"
    timeout -s INT 20m $G $extra --name "$n" --clip-list "$E/cases5/clip_list.json" --staging "$S/vivid" --out "$R3/cases5" --max-items 5
    timeout -s INT 15m $G $extra --name "$n" --pairs --staging "$E/viton_pairs" --out "$R3/pairs" --max-items 10
    true ) > "$E3/render_$n.log" 2>&1
  echo "render $n: $(grep -c '"model"' "$E3/render_$n.log") items"; grep -iE "error|traceback" "$E3/render_$n.log" | tail -3; return 0
}
render base
render f4_2000 "$OUT/pytorch_lora_weights.safetensors"
[ -f "$OUT/checkpoint-1500/pytorch_lora_weights.safetensors" ] && render f4_1500 "$OUT/checkpoint-1500/pytorch_lora_weights.safetensors"

echo "== 7. face / skin / background gate (fails if worse than base on ANY clip) ($(mins))"
CANDS="f4_2000 f4_1500"
for s in cases5 pairs; do
  ( cd "$T" && "$MP" -m eval.preserve_check "$R3/$s" --base base --candidates $CANDS --out "$R3/$s/preserve.md" --model "$SEG" ) \
    || echo "GATE_${s}_EXIT=$?"
done
( cd "$T" && "$PY" - "$R3" <<'PYEOF'
import json, sys
from pathlib import Path
res = {}
for f in sorted(Path(sys.argv[1]).glob("*/preserve_check.json")):
    for c, v in json.loads(f.read_text())["verdict"].items():
        r = res.setdefault(c, {"pass": True, "fails": [], "clips": 0})
        r["pass"] = r["pass"] and v["pass"]
        r["clips"] += v["clips"]
        r["fails"] += [f"{f.parent.name}/{x['clip']}: {', '.join(x['worse_on'])}" for x in v["fails"]]
for c, r in res.items():
    print(f"GATE {c}: {'PASS' if r['pass'] else 'FAIL'} ({r['clips']} clips checked)" + ("" if r["pass"] else f"; worse on {r['fails']}"))
Path(sys.argv[1], "gate_summary.json").write_text(json.dumps(res, indent=2))
PYEOF
) || true

echo "== 8. steadiness / sharpness / colour constancy (5 items) ($(mins))"
( cd "$T" && "$MP" -m eval.stability_report "$R3/cases5" --models base f4_2000 f4_1500 --model "$SEG" --out "$E3/stability.md" ) \
  || echo "STABILITY_EXIT=$?"

echo "== 9. blind videos (garment | camera | A | B): ITEM 5 ONLY first, then all 5; base vs the fxreal4 save that passed the gate (else the final)"
CAND=$( "$PY" -c "
import json, sys
g = json.load(open('$R3/gate_summary.json')) if __import__('os').path.exists('$R3/gate_summary.json') else {}
print(next((c for c in ('f4_2000', 'f4_1500') if g.get(c, {}).get('pass')), 'f4_2000'))" )
( cd "$T" && "$PY" eval/five_test.py video "$AZ_ROOT" --results "$R3/cases5" --lora "$CAND" --seed 29 --name item5_fx4 --only 5 ) \
  || echo "VIDEO5_EXIT=$?"
( cd "$T" && "$PY" eval/five_test.py video "$AZ_ROOT" --results "$R3/cases5" --lora "$CAND" --seed 21 --name five_test_fx4 ) \
  || echo "VIDEO_EXIT=$?"
( cd "$E3" && tar cf "$AZ_ROOT/eval/fx4_review.tar" stability.md results/gate_summary.json results/*/preserve.md render_*.log 2>/dev/null )
echo "FX4_DONE after $(mins): item 5 video $AZ_ROOT/eval/item5_fx4/item5_fx4.mp4, all 5 $AZ_ROOT/eval/five_test_fx4/five_test_fx4.mp4 (keys in the *_KEY folders), LoRA $OUT"
