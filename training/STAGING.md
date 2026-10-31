# Phase I: serving a trained model on staging (and rolling back)

**The live endpoint never changes during this phase.** A trained model goes to a **separate staging endpoint**. Production switches only after the owner approves, and the old model stays one setting away.

## 1. Turn a training result into one servable file (free, CPU Pod)

```bash
python3 scripts/merge_lora.py STAGE1/step_X.pth STAGE2/step_Y.pth dit_merged.pth \
  --check-keys reports/keys_0811.json   # from Phase B step B5b
sha256sum dit_merged.pth
```

- Stage 2 and LHAD save **only LoRA adapters**. The merge folds them into the matching Stage 1 backbone: `W + alpha/r · B·A`.
- `--check-keys` refuses to save unless every tensor name and shape matches the deployed `0811` layout, so the existing loader can read the file.
- If B5b showed that the layouts differ, the merge reports exactly what to rename first.

## 2. Register it on the weights volume

`<volume>/joyai/models/registry.json`:
```json
{
  "azyah-vton-experimental": {"dit": "exp1/dit_merged.pth", "sha256": "…", "notes": "Stage1 2000 + Stage2 1000, Glance+own, commercial manifest"},
  "azyah-vton-candidate":    {"dit": "exp1/dit_merged.pth", "sha256": "…", "notes": "passed eval 2026-.."}
}
```

`joyai-baseline` needs no entry: it's always the deployed upstream `joyai_video_edit_dit_0811.pth`.

## 3. Staging endpoint (paid, owner approval)

- **Endpoint:** a copy of the production settings with:
  - `JOYAI_MODEL_VERSION=azyah-vton-experimental`
  - its own name (e.g. `azyah-vton-staging`)
  - a weights volume that holds the merged file
- **Startup logs:** `[start_upstream] model version: azyah-vton-experimental -> …/dit_merged.pth`.
- **Version reporting:** `ready` and `/health` report `model_version`, so the app and benchmark know which model answered.
- **Unknown or missing version:** the worker **stops** with a clear message instead of quietly serving another model.

## 4. Test, then promote (owner approval)

1. Run `eval/run_eval.py` and the blinded `eval/ab_sheet.py` against the baseline on held-out clips.
2. The phone test goes through a testers-only feature flag in the app (next PR).
3. **Promote:** set `JOYAI_MODEL_VERSION=azyah-vton-candidate` on production.
4. **Roll back:** set it back to `joyai-baseline`, or delete the variable, then save. No image change is needed either way.
