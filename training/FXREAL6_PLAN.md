# fxreal6: a better favourite #2 (owner, 2026-11-15)

Goal: a better version of the model that runs live over WebRTC today (favourite #2, `blend05s065`,
FP8 + compile, 288x512). Not a new look. Status: **planning; nothing trained, nothing spent.**

## The owner's final priorities (2026-11-15), each judged against favourite #2

1. **Garment accuracy**: shape, cut, length, pattern and details of the item.
2. **Colour**: matches the item exactly (colour difference on the garment area against the garment photo).
3. **Motion**: consistent, not jumpy between frames (flicker at or below favourite #2).
4. **Face** redrawn like today, but better: same identity, no paste. Skin tone and texture true.
5. **No hijab or hair invention**, and it keeps working at the live 288x512 FP8 + compile setup.

No special logo work: the existing real data (ViViD videos, VITON-HD photos, the staged sets) is used as
it is. The logo stamp stays on 5% of top and dress inputs only, because it costs nothing, and the logo
leak is reported for information, with no pass bar. The effort goes to motion: every ViViD training
video is used.

What is on the volume (inventory, 2026-11-15): ViViD 2,000 videos (989 tops, 572 dresses, 439 bottoms;
1,599 for training, 401 held out), 25 frames a second, about 4 seconds each, 624x832. VITON-HD in 8
files. No ViViD zip (so no more videos to extract), no own clips, 8 hijab clips, no abayas.

## What must improve (first version of the list)

1. **Garment accuracy**: exact colour, print, logo, pattern, cut and length (shirts, pants, dresses,
   abayas). Plain garments come out plain; the person's own top or logo never shows through.
2. **Motion**: clothes behave like clothes when someone walks past (dress swing, sleeves, trouser legs),
   steady from frame to frame.
3. **Face**: redrawn by the model (no face paste: the owner finds it looks odd), about 90% true to
   the person. Measured as below.
4. **Skin**: same tone, texture and lighting on face, neck, hands and arms; correct new skin where a
   garment uncovers it.
5. **Keep**: favourite #2's background, framing, steadiness and speed. No invented hijab or hair.

## Approach

- Favourite #2 is merged into the base model and frozen. A new rank-64 LoRA is trained on top.
  Files are shipped with the new part at strength 0.5, 0.75 and 1.0 (0 = exactly favourite #2).
- Trained on FLUX.2-klein-base-4B, judged on the live distilled model at 2 steps.
- The model draws each frame on its own. It can learn how fabric hangs in each pose and stay
  consistent between frames; it has no memory of motion beyond an optional previous frame.

## Budget: $100 hard cap (owner, 2026-11-15)

One RTX PRO 6000 at a time, on the training volume (about $2.50 an hour: 40 hours at most).

| Step | Hours | About |
|---|---|---|
| Inventory (CPU Pod) | 0.2 | $0.20 |
| Favourite #2 measured on the test sets | 0.7 | $2 |
| Data build | 4 | $10 |
| Trial, 200 steps | 0.5 | $1.50 |
| Training | 9 to 14 | $23 to $35 |
| Evaluation and videos | 1.5 | $4 |
| **Expected** | **16 to 21** | **$41 to $53** |
| Held back for one repeat of a failed step | | $25 |

Stop rules: the trial's measured step time sets the step count so training never passes $40; no step
starts if it would take the running total past $85. Timings are estimates from the last runs (1.1 s
per step at batch 4, 288x512) scaled for the larger garment photo and the new terms.

Cut to fit $100, and what it costs:

| Cut | Was | Now | Expected loss |
|---|---|---|---|
| GPUs | 4 fast GPUs, 3 to 5 hours | 1 RTX PRO 6000, about a day of pod time over the steps | time only |
| Frame sequences | 2,400 videos x 3 windows | all 1,599 training videos x 2 windows (about 3,200 sequences, 12,800 frames) | fewer moments per video; there are no more videos on the volume |
| Steps | about 7,500 | about 3,200 at batch 16 (3 passes) | less polish on fine print and texture |
| Previous-frame picture | trained in, tested live | dropped | no "sees its previous frame" test; steadiness comes from the sequence term only |
| Second camera size (304x544) | 20% of rows | dropped | none at the live size of 288x512 |
| New test videos | 40 | 24 (9 dresses, 7 tops, 5 bottoms, 3 turns) | pass bar a little less certain |
| Strengths and saves judged | 3 strengths, 3 saves | 2 strengths (0.75, 1.0), 2 saves | a slightly coarser pick |

Kept in full: the logo and plain-garment fixes, the face and skin protection, the head and hair rules.
They come from the data rules and the loss weights, not from volume.

## Sized to the money in the account: $50 (2026-11-15)

The cap stays $100, but the account holds about $21 and the owner tops up to at least $50, and the
RTX PRO 6000 costs $2.70 an hour (the 5090 was not available). So the run is sized to finish inside $50:

| Step | Hours | About |
|---|---|---|
| Favourite #2 baseline (running) | 1 | $2.70 |
| Data build: all 1,599 videos x 1 moment (not 2), 1,000 VITON-HD tops | 3.2 | $8.60 |
| Trial, 200 steps | 0.5 | $1.40 |
| Training: about 6,400 steps at batch 4 (3 passes over about 8,500 rows) | 2.7 to 3.4 | $7 to $9 |
| Evaluation and videos | 1.3 | $3.50 |
| **Expected** | **9 to 10** | **$23 to $26** |

That leaves room for one repeat of the costliest step. No step starts if it would take the total past
$45. What the smaller size costs: one moment per video instead of two (1,600 motion windows, not 3,200),
so less variety of movement per garment; every video and every garment is still used. If the owner tops
up further, the second moment per video and a fourth pass are the first things to add back.

No new training code: the face, skin and hair weights are put into the weight mask by the data step
(face full, skin and hair half), where the trainer's existing colour, texture, edge and pair terms
already work; a 4-frame window is stored as two frame pairs, which the existing pair batching reads.
The run starts from favourite #2 (as fxreal5 did) and the strength dial is a blend back towards it.

## Data

| Part | Size | Notes |
|---|---|---|
| Frame sequences | all 1,599 training videos x 2 windows x 4 frames | frames 4 apart at 25 fps = 0.16 s (the live rate); windows where the clothes move most |
| Stills | about 2,500 VITON-HD tops + existing ViViD stills | studio photos capped at 30% of all rows |
| Logo over own shirt | 5% of top and dress rows | free to make; not a goal of this round |
| Plain answers | 30%+ | captioned with the app's plain wording |
| Skin exposure | 20% | input covers more or less skin than the answer (sleeves, necklines) |
| No change | 15% | input = answer |
| Close-ups | 30% | head and shoulders, so the face has enough pixels to learn from |

Rules: everything outside the garment area is real pixels in both input and answer. A pair is dropped
if head covering, hair or skin differs between them (unless intended), or the garment did not change.
Skin tone is measured per person, in 6 groups, none under about 12% after oversampling.

Captions are the app's live prompt, word for word (`fx6_prompt`). fxreal5's dress captions said
"hijab or head covering" on every dress, and its result kept adding a hijab; the live app says only
"head covering", and fxreal6 trains with exactly that. The word "hijab" never appears.

Missing today: abayas (none), darker skin tones (few), hijab clips (8, used for testing only).

Held out, never trained on: 24 new videos (9 dresses, 7 tops, 5 bottoms, 3 turns), the owner's
existing sets (dress10, set2, cases5, hijab) and a small logo set (information only).

## Recipe

- Camera size 288x512. Garment photo 528x704 (75%), 384x512 (25%).
- LR 5e-5, warm-up then cosine; effective batch 16 (4 sequences of 4 frames); about 3,200 steps
  (set by the trial); a save every 500.
- FluxRT's two live noise levels; the same noise for the 4 frames of a sequence.
- Weights: garment 3x, face 3x, neck/arms/hands 2x, head and hair 2x.
- Terms: garment colour 0.5, garment texture 0.3, edges 0.05, skin colour 0.5, skin texture 0.3,
  face pixels (decoded), identity 0.1, sequence 1.0 (garment, face and skin may change between frames
  only as much as the real frames do).

## Tools (branch `training/fx6-next`)

| Tool | Does | State |
|---|---|---|
| `data_tools/inventory.py` | counts what is on the volume | done, tested |
| `data_tools/fluxrt_prompt.py` `fx6_prompt` | captions = the live app, with the plain wording | done, tested |
| `data_tools/sequences.py` | picks the 4-frame windows with most movement | done, tested |
| `data_tools/logo_stamp.py` | sharp logo, print or stripes on the input garment | done, tested |
| `data_tools/skin_tone.py` | skin-tone groups and balancing | done, tested |
| `eval/face_skin.py` | face match, skin tone, lightness, texture, the pass bar | done, tested |
| `eval/face_skin_report.py`, `eval/logo_set.py` | the face, skin and logo-leak report on rendered clips | done, tested, run end to end on CPU |
| `fluxrt_lora/fx6_baseline.sh` | step 4: favourite #2's own numbers | done; first GPU run is the baseline itself |
| `data_tools/sequences.py` (tool), `build_real.py --windows`, `data_tools/region_weight.py`, `fluxrt_lora/fx6_data.sh` | step 5: the data build | done; tested with a fake re-dresser and end to end on CPU; first GPU run is the build itself |
| trial, training and evaluation script | steps 6 to 8 | next |

## Measures and pass bar (against favourite #2, same garments and poses; truth = the camera frame)

| Priority | Measure | Pass |
|---|---|---|
| 1 Garment accuracy | likeness of the garment area to the garment photo (shape, pattern, details); how well it follows the body | better than favourite #2 on 80%+ of clips; worse on none by more than 2% |
| 2 Colour | colour difference, garment area against the garment photo | lower on average and on 80%+ of clips |
| 3 Motion | garment wobble, colour wander and background flicker between frames | at or below favourite #2 on every set |
| 4 Face | face match (brightness pattern of the face against the camera frame; 1 = the same face) | every clip at least favourite #2's; close-ups 0.90 or more |
| 4 Face | face change between frames beyond the camera's own | no more than favourite #2's |
| 4 Skin | tone difference 3 or less, lightness within 2, fine-detail ratio 0.85 to 1.15 | and no clip worse than favourite #2; no skin-tone group worse than 1.5x the average |
| 5 Head and hair | covering invented or lost | none |
| 5 Live setup | speed at 288x512, FP8 + compile | within 3% of 6.35 base fps |
| Information | logo leak on plain garments | reported, no bar |

The absolute numbers are confirmed once favourite #2's own scores are measured (step 4).

## Order

1. Owner approved the plan with a $100 hard cap (2026-11-15).
2. Tools and tests on branch `training/fx6-next`. No spend.
3. Inventory: `data_tools/inventory.py` on a small CPU Pod with the training volume. Exact counts.
4. Favourite #2 measured on the test sets (GPU Pod). Sets the pass bar.
5. Data build; counts and 20 sample sheets to the owner.
6. 200-step trial; step time gives the final time and cost; owner says go.
7. Full run.
8. Evaluation: tables and labelled side-by-side videos; owner picks the save and the strength.
9. Chosen file to the endpoint volume; `FLUXRT_LORA_PATH` on the test endpoint. Rollback = old path.

The data is research-licence, so the result stays flagged for the owner's use and demos.
