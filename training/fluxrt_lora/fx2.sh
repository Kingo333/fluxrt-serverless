#!/usr/bin/env bash
# FluxRT LoRA "fxreal2" (owner, 2026-11-08): train on FLUX.2-klein-BASE-4B with REAL-photo answers and
# FluxRT's own noise shift, then judge it on the distilled klein-4B at 2 steps (Release #43 frame and
# garment size 288x512) against base FluxRT. The ONLY goal is a better garment change: everything
# outside the garment area is the real pixels in every input (paste-back, also for ViViD), 10% "no
# change" examples, garment-area loss x2 (outside 1x), and an automatic gate that fails the LoRA if it
# is worse than base on face, skin tone or background on ANY clip. One RTX 5090 Pod (RTX PRO 6000 if no 5090), volume
# azyah-vton-train at /workspace. Paid: owner approved ~$3.30 (cap $4.80) on a 5090, ~$7 (cap $10) on a
# PRO 6000. Never touches the live endpoint or the live azyah-joyai volume. Stops its own Pod at the end.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx2.sh -o /workspace/fx2.sh
#   AZ_REF=<SHA> nohup bash /workspace/fx2.sh > /workspace/fx2.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
EXP="${AZ_EXP:-fxreal2}"
GUARD=/workspace/fx2_started
# AZ_NO_STOP=1 keeps the Pod running at the end (to copy files out); stop it by hand afterwards.
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
[ -e "$GUARD" ] && [ "${AZ_FORCE:-0}" != "1" ] && { echo "GUARD: $GUARD exists (already ran); AZ_FORCE=1 to run again"; exit 0; }
touch "$GUARD"
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv
C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E="$AZ_ROOT/eval/fx2"; R="$E/results"
V="$AZ_ROOT/venv-fluxlora/bin"; PY="$V/python"
mkdir -p "$E" "$R" "$AZ_ROOT/reports"
START=$(date +%s)
echo "== 0. space (need ~30 GB free)"
df -h /workspace | tail -1
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 30 ] || { echo "LOW_SPACE: ${FREE_GB} GB free on the volume; owner to free space, then rerun with AZ_FORCE=1"; exit 1; }

echo "== 1. code at $AZ_REF"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -x "$PY" ] || { echo "NO_VENV: $AZ_ROOT/venv-fluxlora missing (P8L's environment); run_train.sh builds it"; }

echo "== 2. real data already on the volume"
for d in vivid vivid_dress_stills vivid_upper_stills vivid_lower_stills; do
  [ -d "$S/$d" ] || { echo "$d: missing"; continue; }
  n=$(find "$S/$d" -mindepth 2 -maxdepth 2 -name meta.json | wc -l)
  live=$(find "$S/$d" -mindepth 2 -maxdepth 2 \( -name live_source.png -o -name live_source.mp4 \) | wc -l)
  echo "$d: items $n, with live-way input $live"
done

echo "== 3. VITON-HD (forgeml/viton_hd, real photos, ~3.4 GB parquet)"
if [ ! -f "$D/raw/viton_hd/.downloaded" ]; then
  # The venv may not exist yet on a brand-new volume: run_train.sh's step 1 builds it; use any hf CLI.
  HF="$V/hf"; [ -x "$HF" ] || { python3 -m pip install -q "huggingface_hub>=0.26"; HF=hf; }
  "$HF" download forgeml/viton_hd --repo-type dataset --local-dir "$D/raw/viton_hd" && touch "$D/raw/viton_hd/.downloaded" \
    || { echo VITON_DOWNLOAD_FAILED; exit 1; }
fi
du -sh "$D/raw/viton_hd"

echo "== 3b. MediaPipe (Apache-2.0) in its own venv: garment-area masks for the ViViD stills, test cases, face gate"
MP="$AZ_ROOT/venv-mp/bin/python"
if [ ! -x "$MP" ] || ! "$MP" -c "import mediapipe, av" 2>/dev/null; then
  python3 -m venv "$AZ_ROOT/venv-mp" && "$AZ_ROOT/venv-mp/bin/pip" install -q mediapipe==0.10.21 av pillow numpy \
    || echo "MEDIAPIPE_INSTALL_FAILED"
fi
"$MP" -c "import mediapipe, av" 2>/dev/null || { echo "NO_MEDIAPIPE: the face/skin/background gate cannot run; stopping before any training"; exit 1; }
( cd "$T" && "$MP" -m data_tools.clothes_mask "$S/vivid_dress_stills" "$S/vivid_upper_stills" "$S/vivid_lower_stills" \
    --model "$AZ_ROOT/selfie_multiclass_256x256.tflite" ) || echo "MASKS_EXIT=$?"

DATA="$D/fluxrt_lora/$EXP"
if [ ! -f "$DATA/dataset_summary.json" ]; then
  echo "== 4. build the real-answer dataset (re-dress with FLUX.2-klein-4B, paste back, filter)"
  "$V/pip" install -q av 2>/dev/null || true
  timeout -s INT 100m "$PY" "$T/fluxrt_lora/build_real.py" --out "$DATA" --viton-dir "$D/raw/viton_hd" \
    --klein "$C/FLUX.2-klein-4B" --vivid-stills "$S/vivid_dress_stills" "$S/vivid_upper_stills" "$S/vivid_lower_stills" \
    --test-out "$E/viton_pairs" --viton-n "${AZ_VITON_N:-2000}" --holdout 300 --test-items 50
  echo "BUILD_EXIT=$?"
fi
cat "$DATA/dataset_summary.json" || { echo NO_DATASET; exit 1; }
echo "build done after $(( ($(date +%s) - START) / 60 )) min"

echo "== 5. train the LoRA on FLUX.2-klein-base-4B (noise shift mu 2.01, 2,000 steps, saves every 250)"
OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"
if [ ! -f "$OUT/pytorch_lora_weights.safetensors" ]; then
  timeout -s INT 120m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=288x512 \
    AZ_MAX_STEPS=${AZ_MAX_STEPS:-2000} AZ_BATCH=4 AZ_LORA_RANK=32 AZ_LR=${AZ_LR:-1e-4} \
    AZ_CKPT_INTERVAL=250 AZ_CKPT_LIMIT=8 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=2.0 bash"
  echo "TRAIN_EXIT=$?"
fi
ls -la "$OUT" | tail -12
[ -f "$OUT/pytorch_lora_weights.safetensors" ] || { echo LORA_MISSING; exit 1; }
cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
echo "train done after $(( ($(date +%s) - START) / 60 )) min"

echo "== 6. test sets: shoe set, no-face items, the owner's five cases (held-out ViViD clips), VITON-HD pairs"
"$PY" -m eval.make_shoe_set "$E/shoe_set" --vivid-staging "$S/vivid" || echo "SHOE_SET_EXIT=$?"
( cd "$T" && "$MP" -m eval.select_test_clips --vivid "$S/vivid" --garments "$T/eval/test_garments.json" \
    --out "$E/cases" --shoe-set "$E/shoe_set" ) || echo "SELECT_EXIT=$?"

echo "== 7. renders: base vs LoRA saves, distilled klein-4B, 2 steps, 288x512, garment 288x512, compact prompt"
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --ref-size 288x512 --steps 2 --prompt-style compact"
render() {  # name lora-file scale (each test set has its own 15-minute limit)
  local n="$1" l="$2" sc="$3" extra=""
  [ -n "$l" ] && extra="--lora $l --lora-scale $sc"
  ( cd "$T"
    timeout -s INT 15m $G $extra --name "$n" --pairs --staging "$E/viton_pairs" --out "$R/pairs" --max-items 50
    timeout -s INT 15m $G $extra --name "$n" --pairs --staging "$E/shoe_set" --out "$R/shoes" --max-items 10
    [ -d "$E/cases/noface" ] && timeout -s INT 15m $G $extra --name "$n" --pairs --staging "$E/cases/noface" --out "$R/noface" --max-items 10
    timeout -s INT 15m $G $extra --name "$n" --cross --staging "$S/vivid" --out "$R/cross" --max-items 8
    [ -f "$E/cases/clip_list.json" ] && timeout -s INT 15m $G $extra --name "$n" --clip-list "$E/cases/clip_list.json" --staging "$S/vivid" --out "$R/cases" --max-items 12
    true
  ) > "$E/render_$n.log" 2>&1
  echo "render $n done: $(grep -c '"model"' "$E/render_$n.log") items"; grep -iE "error|traceback" "$E/render_$n.log" | tail -3
}
LORA="$OUT/pytorch_lora_weights.safetensors"
render base "" 1
render l2000 "$LORA" 1.0
render l2000s07 "$LORA" 0.7
for st in 1500 1000; do
  f="$OUT/checkpoint-$st/pytorch_lora_weights.safetensors"
  [ -f "$f" ] && render "l$st" "$f" 1.0
done
echo "EVAL_DONE (after $(( ($(date +%s) - START) / 60 )) min)"

echo "== 8. reports, side-by-side grids and the blind A/B package"
for s in pairs shoes noface cross cases; do
  [ -d "$R/$s" ] && ( cd "$T" && "$PY" -m eval.run_eval "$R/$s" --out "$R/$s/report.md" ) && echo "--- $s" && head -14 "$R/$s/report.md"
done
for s in pairs shoes noface cross cases; do
  [ -d "$R/$s" ] && ( cd "$T" && "$PY" -m eval.grid "$R/$s" "$E/grid_$s" --models base l2000 l2000s07 l1500 l1000 ) || echo "GRID_${s}_EXIT=$?"
done
for s in pairs shoes noface cross cases; do
  for m in l2000 l2000s07; do
    [ -d "$R/$s" ] && ( cd "$T" && "$PY" -m eval.ab_sheet make "$R/$s" base "$m" "$E/ab_${m}/$s" ) || true
  done
done
echo "== 9. face / skin / background gate (LoRA FAILS if worse than base on ANY clip)"
GATE=PASS
for s in pairs shoes noface cross cases; do
  [ -d "$R/$s" ] || continue
  ( cd "$T" && "$MP" -m eval.preserve_check "$R/$s" --base base --candidates l2000 l2000s07 l1500 l1000 \
      --out "$R/$s/preserve.md" --model "$AZ_ROOT/selfie_multiclass_256x256.tflite" ) || echo "GATE_${s}_EXIT=$?"
done
( cd "$T" && "$PY" - "$R" <<'PYEOF'
import json, sys
from pathlib import Path
res = {}
for f in sorted(Path(sys.argv[1]).glob("*/preserve_check.json")):
    for c, v in json.loads(f.read_text())["verdict"].items():
        r = res.setdefault(c, {"pass": True, "fails": []})
        r["pass"] = r["pass"] and v["pass"]
        r["fails"] += [f"{f.parent.name}/{x['clip']}: {', '.join(x['worse_on'])}" for x in v["fails"]]
for c, r in res.items():
    print(f"GATE {c}: {'PASS' if r['pass'] else 'FAIL'}" + ("" if r["pass"] else f" ({len(r['fails'])} clips, e.g. {r['fails'][:3]})"))
Path(sys.argv[1], "gate_summary.json").write_text(json.dumps(res, indent=2))
PYEOF
) || true
cp -f "$E/cases/cases.json" "$E/cases_picked.json" 2>/dev/null || true
echo "== 10. five blind side-by-side videos for the owner (garment | camera | A | B)"
( cd "$T" && "$PY" -m eval.owner_videos "$R/cases" --gate "$R/gate_summary.json" --out "$AZ_ROOT/eval/owner_videos" ) \
  || echo "OWNER_VIDEOS_EXIT=$?"
ls -la "$AZ_ROOT/eval/owner_videos" 2>/dev/null
cat > "$E/PASS_RULE.txt" <<'TXT'
Owner's pass rule (2026-11-08). The ONLY goal is a better garment change.
A LoRA is adopted only if ALL of these hold:
1. Blind A/B (ab_<candidate>/<set>/review.csv): it wins at least 26 of 40 non-tie picks on the GARMENT.
2. It is NOT worse than base on face likeness, skin tone or background steadiness on ANY clip,
   judged by eye in the A/B AND by the automatic gate (results/gate_summary.json = PASS).
3. It is not worse on any shoe or no-face item.
Otherwise Release #43 FluxRT stays as it is.
TXT
( cd "$E" && tar cf "$AZ_ROOT/eval/fx2_review.tar" PASS_RULE.txt grid_* ab_* cases_picked.json results/*/report.md \
    results/*/preserve.md results/gate_summary.json 2>/dev/null; ls -la "$AZ_ROOT/eval/fx2_review.tar" )
echo "FX2_DONE after $(( ($(date +%s) - START) / 60 )) min"
