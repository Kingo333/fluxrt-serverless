#!/usr/bin/env bash
# fxreal6, step 5 (owner, 2026-11-15): build the training data. Trains nothing.
#   1. frame windows: every ViViD training video (tops, dresses, bottoms), the 2 moments where the clothes
#      move most, 4 frames each, one live frame gap apart (data_tools.sequences)
#   2. the rows: each window as two frame pairs (re-dressed input, real answer, same other garment and seed),
#      the existing ViViD stills, VITON-HD tops, "no change" rows; captions = the app's live prompt
#      (plain wording for plain garment photos); 30% close-ups, 20% long-sleeve inputs over bare arms,
#      5% inputs with a mark on the garment (fluxrt_lora/build_real.py)
#   3. weights and balance: face full weight, skin and hair half weight, rare skin-tone groups repeated
#      (data_tools.region_weight)
#   4. 20 sample sheets for the owner to look at: input | garment | answer | weight mask
# Needs a GPU with >= 30 GB (RTX PRO 6000 or RTX 5090) and the azyah-vton-train volume at /workspace.
# Never touches an endpoint or the azyah-joyai volume. About 4-5 hours; it can be stopped and started
# again (finished windows and a finished dataset are kept).
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx6_data.sh -o /workspace/fx6_data.sh
#   AZ_REF=<SHA> AZ_NO_STOP=1 nohup bash /workspace/fx6_data.sh > /workspace/fx6_data.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
EXP="${AZ_EXP:-fxreal6}"
MAX_MIN="${AZ_MAX_MINUTES:-420}"   # hard stop: 7 hours of this Pod
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E="$AZ_ROOT/eval/fxa"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
DATA="$D/fluxrt_lora/$EXP"
mkdir -p "$E" "$AZ_ROOT/reports"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }
left() { echo $(( MAX_MIN - ($(date +%s) - START) / 60 )); }
echo "== fxreal6 data build ($EXP), hard stop after $MAX_MIN minutes"

echo "== 0. GPU, code, tools"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo NO_GPU; exit 1; }
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -x "$PY" ] && "$PY" -c "import torch, diffusers, pyarrow" 2>/dev/null || { echo "NO_RENDER_VENV: $PY"; exit 1; }
[ -x "$MP" ] && "$MP" -c "import mediapipe, av, pyarrow" 2>/dev/null || { echo "NO_MEDIAPIPE_VENV: $MP"; exit 1; }
[ -f "$D/raw/viton_hd/.downloaded" ] || { echo "NO_VITON: $D/raw/viton_hd"; exit 1; }
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 40 ] || { echo "LOW_SPACE: ${FREE_GB} GB free on the volume (need 40)"; exit 1; }

echo "== 1. frame windows from every ViViD training video ($(mins))"
( cd "$T" && timeout -s INT "$(left)m" "$MP" -m data_tools.sequences "$S/vivid" "$S/vivid_windows" --windows "${AZ_WINDOWS:-2}" \
    --n "${AZ_CLIPS:-0}" --model "$SEG" ) 2>&1 | grep -E "^SEQUENCES|sequence_progress|Error|Traceback" || echo "SEQUENCES_EXIT=$?"
echo "pair folders on disk: $(ls -d "$S"/vivid_windows/*/ 2>/dev/null | wc -l)"
[ "$(ls -d "$S"/vivid_windows/*/ 2>/dev/null | wc -l)" -ge 200 ] || { echo NO_WINDOWS; exit 1; }

if [ ! -f "$DATA/dataset_summary.json" ]; then
  echo "== 2. build the rows ($(mins))"
  timeout -s INT "$(left)m" "$PY" "$T/fluxrt_lora/build_real.py" --out "$DATA" --viton-dir "$D/raw/viton_hd" \
    --klein "$C/FLUX.2-klein-4B" --vivid-stills "$S/vivid_dress_stills" "$S/vivid_upper_stills" "$S/vivid_lower_stills" \
    --windows "$S/vivid_windows" --caption-style live \
    --closeup-frac "${AZ_CLOSEUP:-0.3}" --exposure-frac "${AZ_EXPOSURE:-0.2}" --logo-frac "${AZ_LOGO:-0.05}" \
    --test-out "$AZ_ROOT/eval/fx6_unused_test_pairs" --test-items 0 \
    --viton-n "${AZ_VITON_N:-1500}" --holdout 300 --nochange-frac "${AZ_NOCHANGE:-0.15}" --photo-aug 0.3 \
    --longer-frac "${AZ_LONGER:-0.25}" --longer-ext 0.2 2>&1 | grep -vE "^\s*$" | tail -40
  echo "BUILD_EXIT=${PIPESTATUS[0]}"
fi
[ -f "$DATA/dataset_summary.json" ] || { echo "NO_DATASET (stopped or out of time: start it again, finished windows are kept)"; exit 1; }
"$PY" -c "
import json; s = json.load(open('$DATA/dataset_summary.json'))
keep = ('samples', 'vivid_live', 'viton_kept', 'nochange', 'window_pairs_tried', 'window_pairs_kept', 'window_by_type',
        'closeups', 'logos', 'exposure_kept', 'plain_captions', 'described', 'photo_aug')
print('DATASET ' + json.dumps({k: s.get(k) for k in keep}))"

echo "== 3. weights for face, skin and hair; skin-tone balance ($(mins))"
( cd "$T" && "$MP" -m data_tools.region_weight "$DATA" --model "$SEG" ) 2>&1 | grep -E "^REGIONS|Error|Traceback" || { echo "REGION_WEIGHT_FAILED"; exit 1; }

echo "== 4. 20 sample sheets ($(mins))"
"$PY" - "$DATA" "$E/fx6_samples" <<'PYEOF' || echo "SAMPLES_EXIT=$?"
import io, random, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
data, out = Path(sys.argv[1]) / "data", Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
files = sorted(data.glob("*.parquet"))
rng = random.Random(6)
kinds = {"vivid_win": 10, "_same": 2, "viton_hd": 4, "": 4}   # mostly the new window rows; "_same" = no change
done = 0
for kind, want in kinds.items():
    got = 0
    for f in rng.sample(files, len(files)):
        if got >= want:
            break
        t = pq.read_table(f)
        keys = t.column("key").to_pylist()
        idx = [i for i, k in enumerate(keys) if kind in k]
        if not idx:
            continue
        i = rng.choice(idx)
        row = {c: t.column(c)[i].as_py() for c in ("key", "caption", "person", "garment", "target", "mask")}
        ims = [Image.open(io.BytesIO(row[c]["bytes"])).convert("RGB") for c in ("person", "garment", "target", "mask")]
        h = 512
        ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
        sheet = Image.new("RGB", (sum(im.width for im in ims) + 30, h + 70), "white")
        x = 0
        for im in ims:
            sheet.paste(im, (x, 0)); x += im.width + 10
        d = ImageDraw.Draw(sheet)
        d.text((4, h + 4), f"{row['key']}   input | garment | answer | weights", fill="black")
        cap = row["caption"]
        d.text((4, h + 22), cap[:190], fill="black"); d.text((4, h + 40), cap[190:380], fill="black")
        done += 1; got += 1
        sheet.save(out / f"sample_{done:02d}.jpg", quality=88)
print(f"SAMPLES {done} sheets in {out}")
PYEOF
du -sh "$DATA" | cut -f1 | sed 's/^/dataset size: /'
echo "FX6_DATA_DONE after $(mins): dataset $DATA; sample sheets $E/fx6_samples"
