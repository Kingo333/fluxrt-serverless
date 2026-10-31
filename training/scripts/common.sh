# Shared settings for Azyah-VTON training scripts. Sourced, not run.
# Layout on the training network volume (mounted at /workspace on Pods):
#   /workspace/azyah-vton/
#     JoyAI-Video-Edit/        upstream code at the pinned commit
#     checkpoints/dit/         pretrain_editing_bid_version.pth (Stage 1 start)
#     checkpoints/vae/         VAE
#     checkpoints/MiMo-VL-7B-RL-2508/   text encoder
#     data/toy/                JoyAI ToyData (TAR shards + manifests)
#     outputs/                 training runs (checkpoints, logs)
#     reports/                 key dumps, GPU memory logs, summaries
export AZ_ROOT="${AZ_ROOT:-/workspace/azyah-vton}"
export AZ_UPSTREAM="$AZ_ROOT/JoyAI-Video-Edit"
export AZ_CKPT="$AZ_ROOT/checkpoints"
export AZ_DATA="$AZ_ROOT/data"
export AZ_OUT="$AZ_ROOT/outputs"
export AZ_REPORTS="$AZ_ROOT/reports"
# Same upstream commit as the serving image (Dockerfile.livecam JOYAI_COMMIT).
export JOYAI_COMMIT="${JOYAI_COMMIT:-d875895dd44950caf06f1a2cc4f4601c9eadc5cd}"
export JOYAI_BASE_MODEL="$AZ_CKPT/dit/pretrain_editing_bid_version.pth"
export JOYAI_VAE="$AZ_CKPT/vae"
export JOYAI_TEXT_ENCODER="$AZ_CKPT/MiMo-VL-7B-RL-2508"
# Low-memory Hugging Face downloads (the hf-xet and hf_transfer paths ran out of memory or were missing on small Pods).
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "$AZ_CKPT" "$AZ_DATA" "$AZ_OUT" "$AZ_REPORTS"
# Record which code a run used (review R4): our scripts commit + upstream JoyAI commit.
if [ -n "${AZ_REF:-}" ]; then
  echo "{\"azyah_scripts_ref\": \"$AZ_REF\", \"joyai_commit\": \"$JOYAI_COMMIT\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$AZ_REPORTS/code_refs.jsonl"
fi
