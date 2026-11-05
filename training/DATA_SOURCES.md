# Data and model sources: licences and required notices

| Item | Licence | Notice we must show when a model or app ships |
|---|---|---|
| JoyAI-Video-Edit code and weights (`jd-opensource`, `jdopensource`) | Apache-2.0 | Keep the licence and NOTICE in the image and model card; state changes |
| MiMo-VL-7B-RL-2508 (Xiaomi) | MIT | Keep the MIT copyright notice |
| YuNet face detector (OpenCV Zoo) | MIT | Keep the MIT notice |
| Glance VTON Synthetic Dataset | CC-BY-4.0 | "Glance VTON Synthetic Dataset (2026), https://huggingface.co/datasets/glanceai/VTON-Synthetic-Pairs, CC BY 4.0" on an app credits page and in the model card, if trained on it |
| ViViD | Research only (paper), Hugging Face tag Apache-2.0 (see "ViViD evidence" below) | Not in commercial checkpoints without legal sign-off |
| SAM 2, Grounding DINO (if used for masks) | Apache-2.0 (re-check the checkpoint cards) | Keep the licence notices with the tools |
| JoyAI ToyData | **None stated** (card and tags checked 2026-10-31) | Smoke tests only; never in a released checkpoint |
| Own recordings | Consent forms (see CONSENT_TEMPLATE.md) | Honour withdrawals: remove the person's samples, then retrain or mark checkpoints |
| Ultralytics YOLO | AGPL-3.0 | **Not used** |

## ViViD evidence (checked 2026-11-05)

Why ViViD stays `research_only` until the owner records a legal sign-off:

- **Hugging Face dataset card** (alibaba-yuanjing-aigclab/ViViD): metadata `license: apache-2.0`; the README has nothing else.
- **GitHub repo** (github.com/alibaba-yuanjing-aigclab/ViViD): Apache-2.0 `LICENSE` for the code; no dataset terms.
- **The paper** ("ViViD: Video Virtual Try-on using Diffusion Models", arXiv 2405.11794v2):
  - "We first downloaded a variety of clothing images and related video data from the e-commerce website Net-A-Porter."
  - "The dataset will be available for research purposes."

So the people who released the data say it is for research, and the videos and product photos come from
Net-A-Porter. An Apache-2.0 tag from the uploader may not cover rights in content they did not create.
This is a legal question for the owner (and a lawyer), not a technical one. To clear it, set
`commercial_use_cleared: true` in the model's registry entry with the sign-off recorded (see STAGING.md).
Attribution if used: "ViViD (Fang et al., 2024), alibaba-yuanjing-aigclab/ViViD, Apache-2.0".
