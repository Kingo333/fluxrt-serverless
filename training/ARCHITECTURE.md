# Live try-on architecture (as built, 2026-10-31)

```mermaid
flowchart LR
  subgraph Phone["Phone / browser (azyah-shopping)"]
    CAM[Camera] --> CROP["center-crop + JPEG<br/>JoyAI: 480x840 @24fps q0.85<br/>FluxRT: 288x512 @12 q0.7"]
    PICK[Garment picker] --> PREP["garment photo: white bg, ≤1024px<br/>instruction: category + name +<br/>description / Gemini details"]
    CANVAS[Canvas playback<br/>+ per-frame stats]
  end
  CROP -->|"frame {seq, t_client_ms} base64 JPEG"| WS
  PREP -->|set_reference_image + set_prompt| WS
  subgraph Start["Session start"]
    EF["Supabase Edge Function<br/>live-cam-session-start (v38 live)"] --> CF["Cloudflare Worker<br/>fluxrt-orchestrator (May 22 live)<br/>signs session token"]
  end
  Phone -->|start| EF
  CF -->|ws URL + token| Phone
  WS["WebSocket /ws<br/>RunPod Load Balancer<br/>azyah-runpod-fluxrt-bootstrap"] --> AD
  subgraph Worker["RunPod worker (dual image, LIVECAM_ENGINE)"]
    AD["JoyAI adapter :8765<br/>token check, message bridge,<br/>/ping, model_version"] -->|start / binary JPEG + frame_meta| JOY["JoyAI upstream 127.0.0.1:8080<br/>DiT 16B (0811 or registry version)<br/>2 steps, FP8, SageAttention (5090)"]
    JOY -->|output_frame {source_seq} + JPEG q90| AD
    FLUX["FluxRT server.py<br/>(LIVECAM_ENGINE=fluxrt)"]
  end
  VOL[("Network volume azyah-joyai<br/>EU-RO-1: weights, compile caches,<br/>models/registry.json")] --- JOY
  AD -->|"frame {seq, t_client_ms, server_ms}"| CANVAS
```

**Training side (separate, never touches the above):**
1. Public datasets (Glance, ViViD-R&D) and own clips go through `data_tools` to shards and manifests.
2. A training Pod on volume `azyah-vton-train` runs Stage 1, then Stage 2, then LHAD.
3. `merge_lora.py` produces a registry version, which a staging endpoint serves with `JOYAI_MODEL_VERSION`.
