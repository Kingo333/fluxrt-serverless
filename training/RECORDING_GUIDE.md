# Recording guide: own try-on clips (plan G2)

**Goal:** about **5 people × 10–20 garments** filmed the way Azyah is actually used: a phone or laptop camera, half-body, at home. This is our fine-tuning data **and** our main quality test. Public datasets are all studio full-body, so they can't stand in for it.

## Before anyone is filmed
1. Each person reads and signs `CONSENT_TEMPLATE.md` (lawyer-reviewed first) and gets a **consent ID** (e.g. `C2026-001`). **Adults only.**
2. Record their choices in `consent.csv`: `person_id,consent_id,train,test,commercial` (yes/no each). Use codes like `p01`, never real names.

## Garments
- Use real garments people can wear: start with **tops** (t-shirts, shirts, hoodies, simple jackets), then trousers and dresses.
- Include **prints, logos, text, stripes and textures**. That's where try-on fails most.
- Photograph each garment **flat or on a hanger, on a plain white background, filling the frame, at least 1024 px**, sharp, even light, no person. Save it as `garments/g01.jpg`.
- List it in `garments/garments.csv`: `garment_id,category,name,description` (category `upper`, `lower` or `dress`). The description should name visible details: colour, pattern, logo position, neckline, sleeves.

## Filming each clip (person wearing the garment)
- **Devices:** a phone held at arm's length (front camera) **and** a laptop webcam. Portrait for the phone.
- **Framing:** head to waist for tops, head to knees for trousers and dresses. That's how shoppers see themselves.
- **Light:** two setups per person, a bright room and normal evening indoor light. Avoid a strong window behind the person.
- **Length:** about 20 seconds at 24–30 fps, at least 720p, no beauty filters or effects.
- **Movement script** (same every time):
  1. 3 s still, facing the camera
  2. slow turn left and back
  3. slow turn right and back
  4. raise both arms, lower them
  5. step closer, then back
  6. 3 s still
- One garment per clip. Save as `clips/p01/g01/phone_bright.mp4`, `clips/p01/g01/webcam_evening.mp4`, …

## Test set
Pick **1–2 whole people** and **3–5 whole garments** for test only, before training. Don't film test people in training garments, or vice versa. The importer flags and blocks any overlap.

## Import
```bash
python3 -m data_tools.ingest_own RAW_DIR STAGING_DIR --test-people p05 --test-garments g17 g18 g19
python3 -m data_tools.masks STAGING_DIR --only own                 # GPU Pod
python3 -m data_tools.pseudo_source STAGING_DIR --dit … --vae … --text-encoder …   # GPU Pod, paid
python3 -m data_tools.pack STAGING_DIR SHARDS_DIR --name own
python3 -m data_tools.validate SHARDS_DIR/*.tar --catalog SHARDS_DIR/own.catalog.json
```
People who said `commercial=no` are tagged `own_research` and kept out of commercial training automatically.

## Storage and privacy
- Keep raw clips on the training volume or the private Hugging Face data repo only. **Never** on GitHub or personal cloud drives.
- If someone withdraws: delete their folder and staged pairs, re-pack, and note which checkpoints were trained before the withdrawal.
