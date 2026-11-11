#!/usr/bin/env bash
# FluxRT LoRA "fxreal5" (owner, 2026-11-11): dresses and abayas. CONTINUES favourite #2 (blend05s065, the exact
# fxreal3+fxreal4 merge at strength 0.65 that runs live) instead of starting from scratch:
#   - the owner's five priorities: COLOUR (garment colour term), DETAIL (edge term 0.05, garment area 3x, bigger
#     garment photo), FABRIC (garment contrast term: shine, sheen, print contrast), MOVEMENT (frames where the
#     dress moves most, several poses per dress) and CONSISTENCY (same-clip pairs with the steady-colour term)
#   - mix (owner): dresses 57% / tops and bottoms 43% (dresses ~35% more than tops and bottoms together):
#     build_real.py --dress-share builds the dresses first (every ViViD dress train clip at the frames where the
#     dress MOVES most, plus the dress stills and dress pairs we already had; half the clips with a
#     T-shirt-and-trousers INPUT, the live case) and then fills tops and bottoms to 43% (ViViD stills, fxreal3's
#     pairs, VITON-HD incl. fxreal4's longer-to-shorter tops); captions carry ViViD's garment description
#   - rank 64 (= favourite #2), learning rate halved (5e-5), 1,500 steps, saves every 250
#   - garment photo size AZ_REF_SIZE: the live FLUXRT_REF_WIDTH x HEIGHT. Favourite #2 was trained and served at
#     288x512; Option A (Release #52, 2026-11-11) serves 576x768. Train at 576x768 only if the owner keeps it.
#   - the live dress prompt names the shopper's own hijab or head covering (data_tools.fluxrt_prompt, = the app)
#   - garment photo sizes (owner, 2026-11-11): AZ_REF_SIZES="576x768:2,432x576:1,288x512:1" trains on several
#     sizes, one per batch (patches/flux2-klein-refsizes.patch), so fxreal5 works at whatever FLUXRT_REF size runs
#     live; the test then also renders favourite #2 and fxreal5 at the other sizes on dress10 and set2 (PRIORITY
#     lines per size, extra panels in those videos). Unset = AZ_REF_SIZE only, as before.
# Then renders on 10 NEW held-out dress clips (dress10, auto-described), the owner's Step B set (set2, his own
# descriptions; two long black maxi dresses), his 5 cases (regression) and the HIJAB set (hijab-wearing people
# trying on a dress: own clips from eval/own_clips plus checked free stock clips, eval.hijab_set; her hijab
# must stay, measured by the gate's head / head_lost numbers): original / favourite #2 / fxreal5
# (final and step 1000) at the same garment size, the face/skin/background gate, the steadiness table, the
# five-priority garment report (colour, detail, fabric, movement, consistency: eval.garment_report) and one
# LABELLED video per set: garment | camera | favourite #2 (today) | fxreal5.
# Needs a GPU with >= 30 GB (RTX PRO 6000 recommended) and the azyah-vton-train volume at /workspace.
# Never touches the live endpoint or the azyah-joyai volume. About 2-3 h on an RTX PRO 6000 ($2.49/h): about $6-8, cap $12.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx5.sh -o /workspace/fx5.sh
#   AZ_REF=<SHA> AZ_REF_SIZE=576x768 AZ_REF_SIZES="576x768:2,432x576:1,288x512:1" AZ_NO_STOP=1 \
#     nohup bash /workspace/fx5.sh > /workspace/fx5.log 2>&1 &
#   (AZ_REF_SIZE=288x512 if the owner goes back to the old garment photo size)
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
EXP="${AZ_EXP:-fxreal5}"
REF_SIZE="${AZ_REF_SIZE:-288x512}"
case "$REF_SIZE" in *x*) ;; *) echo "BAD_AZ_REF_SIZE: $REF_SIZE (e.g. 384x512 or 576x768)"; exit 1 ;; esac
TAG="r${REF_SIZE%%x*}"
EXTRA_SIZES=""  # the other trained garment-photo sizes, rendered too
for spec in $(echo "${AZ_REF_SIZES:-}" | tr ',' ' '); do
  sz="${spec%%:*}"
  case "$sz" in [0-9]*x[0-9]*) ;; *) echo "BAD_AZ_REF_SIZES: $AZ_REF_SIZES (e.g. 576x768:2,432x576:1,288x512:1)"; exit 1 ;; esac
  [ "$sz" != "$REF_SIZE" ] && EXTRA_SIZES="$EXTRA_SIZES $sz"
done
GUARD=/workspace/fx5_started
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
[ -e "$GUARD" ] && [ "${AZ_FORCE:-0}" != "1" ] && { echo "GUARD: $GUARD exists (already ran); AZ_FORCE=1 to run again"; exit 0; }
touch "$GUARD"
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E2="$AZ_ROOT/eval/fx2"; E="$AZ_ROOT/eval/fxa"; R="$E/results"
V="$AZ_ROOT/venv-fluxlora/bin"; PY="$V/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
INIT="$AZ_ROOT/outputs/fluxrt_lora/blend05s065"
mkdir -p "$E" "$R" "$AZ_ROOT/reports"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }
echo "== fxreal5: garment photo $REF_SIZE (tag $TAG), trained sizes ${AZ_REF_SIZES:-$REF_SIZE only}, continuing favourite #2"

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
[ -f "$INIT/pytorch_lora_weights.safetensors" ] || { echo "NO_FAVOURITE2: $INIT/pytorch_lora_weights.safetensors (fxm.sh made it)"; exit 1; }
[ -f "$D/raw/viton_hd/.downloaded" ] || { echo "NO_VITON: $D/raw/viton_hd not downloaded (fx2 does that)"; exit 1; }
sha256sum "$INIT/pytorch_lora_weights.safetensors"

echo "== 2. MediaPipe venv"
if [ ! -x "$MP" ] || ! "$MP" -c "import mediapipe, av" 2>/dev/null; then
  python3 -m venv "$AZ_ROOT/venv-mp" && "$AZ_ROOT/venv-mp/bin/pip" install -q mediapipe==0.10.21 av pillow numpy || true
fi
"$MP" -c "import pyarrow" 2>/dev/null || "$AZ_ROOT/venv-mp/bin/pip" install -q pyarrow
"$MP" -c "import mediapipe, av, pyarrow" 2>/dev/null || { echo "NO_MEDIAPIPE"; exit 1; }

echo "== 3. dress frames where the dress moves: every ViViD dress train clip ($(mins))"
( cd "$T" && "$MP" -m data_tools.vivid_motion "$S/vivid" "$S/vivid_dress_motion" --category dress \
    --n "${AZ_MOTION_N:-0}" --singles "${AZ_SINGLES:-2}" --model "$SEG" ) || echo "MOTION_EXIT=$?"
echo "motion clips on disk: $(ls -d "$S"/vivid_dress_motion/*/ 2>/dev/null | wc -l)"
[ -d "$S/vivid_pairs" ] || ( cd "$T" && "$MP" -m data_tools.vivid_pairs "$S/vivid" "$S/vivid_pairs" --n 300 --model "$SEG" ) || echo "PAIRS_EXIT=$?"

DATA="$D/fluxrt_lora/$EXP"
if [ ! -f "$DATA/dataset_summary.json" ]; then
  echo "== 4. build the dataset: dresses first (${AZ_DRESS_SHARE:-0.57} of the rows), then tops and bottoms ($(mins))"
  timeout -s INT 150m "$PY" "$T/fluxrt_lora/build_real.py" --out "$DATA" --viton-dir "$D/raw/viton_hd" \
    --klein "$C/FLUX.2-klein-4B" --vivid-stills "$S/vivid_dress_stills" "$S/vivid_upper_stills" "$S/vivid_lower_stills" \
    --vivid-pairs "$S/vivid_pairs" --motion "$S/vivid_dress_motion" --dress-outfit-frac "${AZ_OUTFIT:-0.5}" \
    --dress-share "${AZ_DRESS_SHARE:-0.57}" --other-live-max "${AZ_OTHER_LIVE_MAX:--1}" \
    --test-out "$AZ_ROOT/eval/fx5_unused_test_pairs" --test-items 0 \
    --viton-n "${AZ_VITON_N:-1500}" --holdout 300 --nochange-frac "${AZ_NOCHANGE:-0.2}" --photo-aug 0.3 \
    --longer-frac "${AZ_LONGER:-0.25}" --longer-ext 0.2
  echo "BUILD_EXIT=$?"
fi
cat "$DATA/dataset_summary.json" || { echo NO_DATASET; exit 1; }
"$PY" -c "
import json; s = json.load(open('$DATA/dataset_summary.json'))
print(f\"DRESS_SHARE {s.get('dress_share')} ({s.get('dress_rows')} dress rows, {s.get('other_rows')} tops/bottoms rows, of {s.get('samples')}); described rows {s.get('described')}\")
if abs((s.get('dress_share') or 0) - ${AZ_DRESS_SHARE:-0.57}) > 0.05: print('WARNING_DRESS_SHARE_OFF_TARGET: not enough rows of one kind; the run continues')"

echo "== 4b. face counts 2x: add the face to every sample's weight mask ($(mins))"
( cd "$T" && "$MP" -m data_tools.face_weight "$DATA" --model "$SEG" ) || { echo "FACE_WEIGHT_FAILED"; exit 1; }

echo "== 5. train fxreal5 from favourite #2 on FLUX.2-klein-base-4B: ${AZ_MAX_STEPS:-1500} steps, saves every 250 ($(mins))"
OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"
if [ ! -f "$OUT/pytorch_lora_weights.safetensors" ]; then
  timeout -s INT 200m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=$REF_SIZE \
    AZ_MAX_STEPS=${AZ_MAX_STEPS:-1500} AZ_BATCH=4 AZ_LORA_RANK=64 AZ_LR=${AZ_LR:-5e-5} \
    AZ_CKPT_INTERVAL=250 AZ_CKPT_LIMIT=8 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=${AZ_MASK_WEIGHT:-3.0} \
    AZ_PAIR_COLUMN=pair AZ_PAIR_WEIGHT=${AZ_PAIR_WEIGHT:-1.0} AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.05} \
    AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5} AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3} \
    AZ_REF_SIZES=${AZ_REF_SIZES:-} AZ_INIT_LORA=$INIT bash"
  echo "TRAIN_EXIT=$?"
fi
grep -h "LoRA initialised from" "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | tail -1 || true
grep -h "^REF_SIZES" "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | tail -1 || true
ls -la "$OUT" | tail -12
[ -f "$OUT/pytorch_lora_weights.safetensors" ] || { echo LORA_MISSING; exit 1; }
cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
grep -oE "(loss|detail|pair|colour|texture)=[0-9.e+-]+" "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | tail -10
echo "train done after $(mins)"

echo "== 6. eval sets: 10 new held-out dress clips (auto-described), the owner's set2, his 5 cases ($(mins))"
[ -f "$E/dress10.json" ] || ( cd "$T" && "$PY" -m eval.step_a pick "$AZ_ROOT" --set dress10 --n 10 --categories dress --auto-describe ) \
  || echo "PICK_EXIT=$?"
[ -f "$E/cases5.json" ] || cp "$E2/cases5/clip_list.json" "$E/cases5.json" 2>/dev/null || echo "(no cases5 clip list)"
echo "== 6b. hijab set: hijab-wearing people trying on a dress; her own hijab must stay ($(mins))"
[ -s "$E/hijab.json" ] && [ "$(cat "$E/hijab.json")" != "[]" ] || ( cd "$T" && "$MP" -m eval.hijab_set "$AZ_ROOT" \
    --out "$E/hijab.json" --own "$AZ_ROOT/eval/own_clips" --clips-dir "$AZ_ROOT/eval/hijab_clips" \
    --refs-from "$E/set2.json" "$E/dress10.json" --n "${AZ_HIJAB_N:-3}" --model "$SEG" ) 2>&1 | grep -E "HIJAB_CLIPS|NO_HIJAB_CLIP|download failed|only .* of the best window" || echo "HIJAB_SET_EXIT=$?"
[ "$(cat "$E/hijab.json" 2>/dev/null)" = "[]" ] && rm -f "$E/hijab.json"
SETS=""
for s in dress10 set2 cases5 hijab; do [ -f "$E/$s.json" ] && SETS="$SETS $s"; done
echo "sets:$SETS"

echo "== 7. renders at garment size $REF_SIZE: original, favourite #2, fxreal5 (final, step 1000) ($(mins))"
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --steps 2 --prompt-style compact"
render() {  # set name [lora-file] [garment-photo size]
  local s="$1" n="$2" extra=""
  [ -n "${3:-}" ] && extra="--lora $3 --lora-scale 1.0"
  ( cd "$T" && timeout -s INT 25m $G --ref-size "${4:-$REF_SIZE}" $extra --name "$n" --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 10 ) \
    > "$E/render_fx5_${s}_$n.log" 2>&1
  echo "render $s/$n: $(grep -c '"model"' "$E/render_fx5_${s}_$n.log") clips ($(mins))"
  grep -iE "error|traceback" "$E/render_fx5_${s}_$n.log" | tail -3; return 0
}
for s in $SETS; do
  render "$s" "base_$TAG"
  render "$s" "fav2_$TAG" "$INIT/pytorch_lora_weights.safetensors"
  render "$s" "fx5_$TAG" "$OUT/pytorch_lora_weights.safetensors"
  [ -f "$OUT/checkpoint-1000/pytorch_lora_weights.safetensors" ] && render "$s" "fx5s1000_$TAG" "$OUT/checkpoint-1000/pytorch_lora_weights.safetensors"
done
SIZE_SETS=""
for s in dress10 set2; do case " $SETS " in *" $s "*) SIZE_SETS="$SIZE_SETS $s" ;; esac; done
if [ -n "$EXTRA_SIZES" ]; then
  echo "== 7b. the other garment-photo sizes:$EXTRA_SIZES; favourite #2 and fxreal5 on$SIZE_SETS ($(mins))"
  for sz in $EXTRA_SIZES; do
    for s in $SIZE_SETS; do
      render "$s" "fav2_r${sz%%x*}" "$INIT/pytorch_lora_weights.safetensors" "$sz"
      render "$s" "fx5_r${sz%%x*}" "$OUT/pytorch_lora_weights.safetensors" "$sz"
    done
  done
fi

echo "== 8. face / skin / background gate vs the original, and steadiness ($(mins))"
CANDS="fav2_$TAG fx5_$TAG fx5s1000_$TAG"
for s in $SETS; do
  ( cd "$T" && "$MP" -m eval.preserve_check "$R/$s" --base "base_$TAG" --candidates $CANDS --out "$E/preserve_fx5_$s.md" --model "$SEG" ) \
    || echo "GATE_${s}_EXIT=$?"
  ( cd "$T" && "$MP" -m eval.stability_report "$R/$s" --models "base_$TAG" $CANDS --model "$SEG" --out "$E/stability_fx5_$s.md" ) \
    || echo "STABILITY_${s}_EXIT=$?"
done
( cd "$T" && "$PY" - "$R" $SETS <<'PYEOF'
import json, sys
from pathlib import Path
res = {}
for s in sys.argv[2:]:
    f = Path(sys.argv[1], s, "preserve_check.json")
    if not f.exists():
        continue
    for c, v in json.loads(f.read_text())["verdict"].items():
        r = res.setdefault(c, {"pass": True, "fails": [], "clips": 0})
        r["pass"] = r["pass"] and v["pass"]
        r["clips"] += v["clips"]
        r["fails"] += [f"{s}/{x['clip']}: {', '.join(x['worse_on'])}" for x in v["fails"]]
for c, r in res.items():
    print(f"GATE {c}: {'PASS' if r['pass'] else 'FAIL'} ({r['clips']} clips)" + ("" if r["pass"] else f"; worse on {r['fails']}"))
Path(sys.argv[1], "fx5_gate_summary.json").write_text(json.dumps(res, indent=2))
f = Path(sys.argv[1], "hijab", "preserve_check.json")
if f.exists():
    for clip, sc in json.loads(f.read_text())["per_clip"].items():
        base = next((m for m in sc if m.startswith("base_")), None)
        if base is None or (sc[base].get("head_covered") or 0) < 0.5:
            continue
        pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"
        num = lambda v: "-" if v is None else f"{v:.1f}"
        for m, v in sc.items():
            if m != base:
                print(f"HIJAB {clip} {m}: hijab turned into hair/face/skin {pct(v.get('head_lost'))} "
                      f"(original FluxRT {pct(sc[base].get('head_lost'))}), hijab change {num(v.get('head'))} "
                      f"(original {num(sc[base].get('head'))}) on a 0-255 scale")
PYEOF
) || true

echo "== 8b. the owner's five priorities vs favourite #2: colour, detail, fabric, movement, consistency ($(mins))"
for s in $SETS; do
  ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models $CANDS --model "$SEG" ) > "$E/garment_seg_fx5_$s.log" 2>&1 \
    || echo "GARMENT_SEG_${s}_EXIT=$?"
  echo "-- $s"
  ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models $CANDS --baseline "fav2_$TAG" --out "$E/garment_fx5_$s.md" ) \
    2>&1 | grep -E "^PRIORITY|DINO" || echo "GARMENT_SCORE_${s}_EXIT=$?"
done

for sz in $EXTRA_SIZES; do
  t="r${sz%%x*}"
  for s in $SIZE_SETS; do
    ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models "fav2_$t" "fx5_$t" --model "$SEG" ) \
      > "$E/garment_seg_fx5_${s}_$t.log" 2>&1 || echo "GARMENT_SEG_${s}_${t}_EXIT=$?"
    echo "-- $s, garment photo $sz"
    ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models "fav2_$t" "fx5_$t" --baseline "fav2_$t" \
        --out "$E/garment_fx5_${s}_$t.md" --json "$R/$s/garment_report_$t.json" ) 2>&1 | grep -E "^PRIORITY" \
      || echo "GARMENT_SCORE_${s}_${t}_EXIT=$?"
  done
done

echo "== 9. labelled videos: garment | camera | favourite #2 (today) | fxreal5 [| fxreal5 at the other sizes] ($(mins))"
for s in $SETS; do
  PANELS=("fav2_$TAG=favourite #2 (today)" "fx5_$TAG=fxreal5")
  case " $SIZE_SETS " in *" $s "*) for sz in $EXTRA_SIZES; do PANELS+=("fx5_r${sz%%x*}=fxreal5, photo $sz"); done ;; esac
  ( cd "$T" && "$PY" -m eval.step_a video "$AZ_ROOT" --set "$s" --cols $(( ${#PANELS[@]} + 2 )) --no-legend \
      --panels "${PANELS[@]}" ) || echo "VIDEO_${s}_EXIT=$?"
  [ -f "$E/${s}_labelled.mp4" ] && mv -f "$E/${s}_labelled.mp4" "$E/fx5_${s}_${TAG}.mp4"
done
( cd "$E" && tar cf "$AZ_ROOT/eval/fx5_review.tar" dress10.json $( [ -f hijab.json ] && echo hijab.json ) preserve_fx5_*.md stability_fx5_*.md garment_fx5_*.md \
    results/fx5_gate_summary.json results/*/garment_report*.json render_fx5_*.log 2>/dev/null )
echo "FX5_DONE after $(mins): videos $(ls "$E"/fx5_*_"$TAG".mp4 2>/dev/null | tr '\n' ' ')"
echo "LoRA $OUT/pytorch_lora_weights.safetensors (+ selection.json); reports $AZ_ROOT/eval/fx5_review.tar"
