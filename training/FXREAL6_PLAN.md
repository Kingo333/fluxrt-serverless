# fxreal6: a better favourite #2 (owner, 2026-11-15)

Goal: a better version of the model that runs live over WebRTC today (favourite #2, `blend05s065`,
FP8 + compile, 288x512). Not a new look. Status: **planning; nothing trained, nothing spent.**

## What must improve

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

## Data

| Part | Size | Notes |
|---|---|---|
| Frame sequences | about 2,400 videos x 3 windows x 4 frames | frames about 0.17 s apart (the live rate); windows where the clothes move most; 10%+ back and side views |
| Stills | about 4,000 VITON-HD tops + existing ViViD stills | studio photos capped at 30% of all rows |
| Logo over own shirt | 25% of top and dress rows | input garment carries a sharp logo, print or stripes; the answer is the real garment |
| Plain answers | 30%+ | captioned with the app's plain wording |
| Skin exposure | 20% | input covers more or less skin than the answer (sleeves, necklines) |
| No change | 15% | input = answer |
| Close-ups | 30% | head and shoulders, so the face has enough pixels to learn from |

Rules: everything outside the garment area is real pixels in both input and answer. A pair is dropped
if head covering, hair or skin differs between them (unless intended), or the garment did not change.
Head-covering words appear on real hijab rows and on half of the uncovered rows (the answer keeps the
hair), so the words never mean "add one". Skin tone is measured per person, in 6 groups, none under
about 12% after oversampling.

Missing today: abayas (none), darker skin tones (few), real logo shirts on camera, hijab clips.

Held out, never trained on: 40 new videos (15 dresses, 12 tops, 8 bottoms, 5 turns), the owner's
existing sets, a logo set and a close-up face set.

## Recipe

- Camera size 288x512 (80%), 304x544 (20%). Garment photo 528x704 (75%), 384x512 (25%).
- LR 5e-5, warm-up then cosine; effective batch 16 (4 sequences of 4 frames); about 7,500 steps;
  a save every 500.
- FluxRT's two live noise levels; the same noise for the 4 frames of a sequence.
- Weights: garment 3x, face 3x, neck/arms/hands 2x, head and hair 2x.
- Terms: garment colour 0.5, garment texture 0.3, edges 0.05, skin colour 0.5, skin texture 0.3,
  face pixels (decoded), identity 0.1, sequence 1.0 (garment, face and skin may change between frames
  only as much as the real frames do).
- Previous frame: half the rows get an earlier frame of the same video as a third picture.

## Measures and pass bar (against favourite #2, same garments and poses; truth = the camera frame)

| Measure | Pass |
|---|---|
| Face match | close-ups (face 80 px or wider): 90% or more of the score the camera gets against itself one frame later; full body: better than favourite #2 on every clip |
| Face shape and expression | landmark shift 3% of eye distance or less |
| Face drift between frames | no more than the camera's own + 10% |
| Skin tone | colour difference 3 or less, lightness within 2, no skin-tone group worse than 1.5x the average |
| Skin texture | fine-detail ratio 0.85 to 1.15 |
| New skin (sleeves, necklines) | within 4 of the person's own tone |
| Garment colour | better than favourite #2 on 80%+ of clips |
| Print, logo, pattern | 10%+ better |
| Logo leak on plain garments | 5% of frames or fewer |
| Flicker, background, sharpness | no worse on any clip |
| Head covering | none invented, none lost |
| Speed | within 3% of 6.35 base fps (FP8 + compile) |

The absolute numbers are confirmed once favourite #2's own scores are measured (step 4).

## Order

1. Owner approves the plan and a cost cap (suggested $250; the trial gives the real quote).
2. Tools and tests on branch `training/fx6-next`. No spend.
3. Inventory: `data_tools/inventory.py` on a small CPU Pod with the training volume. Exact counts.
4. Favourite #2 measured on the test sets (GPU Pod). Sets the pass bar.
5. Data build; counts and 20 sample sheets to the owner.
6. 200-step trial; step time gives the final time and cost; owner says go.
7. Full run.
8. Evaluation: tables and labelled side-by-side videos; owner picks the save and the strength.
9. Chosen file to the endpoint volume; `FLUXRT_LORA_PATH` on the test endpoint. Rollback = old path.

The data is research-licence, so the result stays flagged for the owner's use and demos.
