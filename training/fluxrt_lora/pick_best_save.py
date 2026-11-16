"""Pick the best save of a training run against favourite #2 (fxreal6b, owner 2026-11-16: keep the best save, not
the last one).

  python -m fluxrt_lora.pick_best_save EVAL_DIR --baseline fav2_r528 [--weights cases5=2,fx6test=1,dress10=1]

EVAL_DIR/<save>/ holds meta.json ({"step", "name", "weights", "sha256"}) and, per test set, <set>/garment.json
(eval.garment_report score), <set>/face_skin.json (eval.face_skin_report), <set>/preserve_check.json
(eval.preserve_check: the GATE) and <set>/stability.json (eval.stability_report). A save is judged once all its
sets are there (marker EVAL_DIR/<save>/.scored).

Per set, every number is turned into a relative gain over favourite #2 (+ = better, capped at +-50% so one odd
clip cannot decide alone) and averaged:
  garment      colour_de, fabric, follow_px, jitter, flicker_de, length_err (lower better), likeness (higher)
  face, skin   face_match (higher), skin_de, face_drift (lower), |texture - 1| (lower: 1 = camera's detail)
  steadiness   wobble_garment, colour_wander (lower), sharpness (higher)
The run's score is the weighted mean over the sets (the owner's 5 real cases count double). Saves are ranked by
the number of sets where the GATE passes, then by the score. Writes EVAL_DIR/best.json and prints one SAVE line
per save and a BEST line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CAP = 0.5
# metric: (lower is better, floor of the denominator: the size of a change that counts as "100%")
GARMENT = {"colour_de": (True, 1.0), "fabric": (True, 0.05), "follow_px": (True, 0.5), "jitter": (True, 0.01),
           "flicker_de": (True, 0.5), "length_err": (True, 0.03), "likeness": (False, 0.05)}
FACE = {"face_match": (False, 0.05), "skin_de": (True, 0.5), "face_drift": (True, 0.1), "texture_off": (True, 0.05)}
STEADY = {"wobble_garment": (True, 0.005), "colour_wander": (True, 0.5), "sharpness": (False, 0.05)}
DEFAULT_WEIGHTS = {"cases5": 2.0, "fx6test": 1.0, "dress10": 1.0}


def gain(cand, base, lower: bool, floor: float):
    """Relative gain of cand over base (+ = better), capped at +-CAP; None when either is missing."""
    if cand is None or base is None:
        return None
    d = (base - cand) if lower else (cand - base)
    g = d / max(abs(base), floor)
    return max(-CAP, min(CAP, g))


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def set_numbers(d: Path, name: str, base: str) -> dict:
    """{metric: (cand, base)} for one set of one save, plus "gate" and "face_gate" (True/False/None)."""
    out: dict = {}
    g = _load(d / "garment.json") or {}
    for row in g.get("scorecard", []):
        if row.get("model") == name and row.get("baseline") == base and row.get("metric") in GARMENT:
            out[row["metric"]] = (row.get("mean"), row.get("baseline_mean"))
    f = _load(d / "face_skin.json") or {}
    av = f.get("average", {})
    a, b = av.get(name) or {}, av.get(base) or {}
    for k in ("face_match", "skin_de", "face_drift"):
        out[k] = (a.get(k), b.get(k))
    ta, tb = a.get("texture"), b.get("texture")
    out["texture_off"] = (None if ta is None else abs(ta - 1), None if tb is None else abs(tb - 1))
    st = _load(d / "stability.json") or {}
    for k in STEADY:
        vals = {m: [clip[m][k] for clip in st.values() if isinstance(clip, dict) and isinstance(clip.get(m), dict)
                    and clip[m].get(k) is not None] for m in (name, base)}
        out[k] = tuple((sum(v) / len(v)) if v else None for v in (vals[name], vals[base]))
    pc = _load(d / "preserve_check.json") or {}
    out["gate"] = (pc.get("verdict", {}).get(name) or {}).get("pass")
    out["face_gate"] = (f.get("verdict", {}).get(name) or {}).get("pass")
    return out


def set_score(nums: dict):
    gains = []
    for table in (GARMENT, FACE, STEADY):
        for k, (lower, floor) in table.items():
            c, b = nums.get(k, (None, None))
            x = gain(c, b, lower, floor)
            if x is not None:
                gains.append(x)
    return (sum(gains) / len(gains)) if gains else None


def judge(root: Path, base: str, weights: dict) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / ".scored").exists()):
        meta = _load(d / "meta.json") or {}
        name = meta.get("name")
        if not name:
            continue
        per_set, total, wsum, gates, face_gates = {}, 0.0, 0.0, 0, 0
        for s, w in weights.items():
            if not (d / s).is_dir():
                continue
            nums = set_numbers(d / s, name, base)
            sc = set_score(nums)
            per_set[s] = {"score": None if sc is None else round(sc, 4), "gate": nums["gate"],
                          "face_gate": nums["face_gate"],
                          "colour": nums.get("colour_de"), "fabric": nums.get("fabric"),
                          "length_err": nums.get("length_err"), "face_match": nums.get("face_match"),
                          "skin_de": nums.get("skin_de"), "face_drift": nums.get("face_drift")}
            if sc is not None:
                total += w * sc
                wsum += w
            gates += nums["gate"] is True
            face_gates += nums["face_gate"] is True
        rows.append({"save": d.name, "step": meta.get("step"), "name": name, "weights": meta.get("weights"),
                     "sha256": meta.get("sha256"), "score": round(total / wsum, 4) if wsum else None,
                     "gate_sets": gates, "face_gate_sets": face_gates, "sets": len(per_set), "per_set": per_set})
    return rows


def rank_key(r: dict):
    return (r["gate_sets"], r["score"] if r["score"] is not None else -1e9)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--baseline", required=True, help="favourite #2's render name, e.g. fav2_r528")
    ap.add_argument("--weights", default=",".join(f"{k}={v:g}" for k, v in DEFAULT_WEIGHTS.items()))
    a = ap.parse_args(argv)
    weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    root = Path(a.eval_dir)
    rows = judge(root, a.baseline, weights)
    fmt = lambda x: "-" if x is None else f"{x * 100:+.1f}%"  # noqa: E731
    for r in sorted(rows, key=lambda r: (r["step"] if r["step"] is not None else -1)):
        sets = "; ".join(f"{s} {fmt(v['score'])}{' GATE ok' if v['gate'] else ' GATE fail'}"
                         + (f", colour {v['colour'][0]} vs {v['colour'][1]}" if v["colour"] and None not in v["colour"] else "")
                         + (f", length {v['length_err'][0]} vs {v['length_err'][1]}" if v["length_err"] and None not in v["length_err"] else "")
                         for s, v in r["per_set"].items())
        print(f"SAVE step {r['step']} ({r['name']}): score {fmt(r['score'])} vs favourite #2; GATE {r['gate_sets']}/{r['sets']} sets; {sets}")
    best = max(rows, key=rank_key) if rows else None
    result = {"baseline": a.baseline, "weights": weights, "saves": rows, "best": best}
    (root / "best.json").write_text(json.dumps(result, indent=1))
    if best:
        print(f"BEST step {best['step']} ({best['name']}): score {fmt(best['score'])}, GATE {best['gate_sets']}/{best['sets']} sets"
              + ("; beats favourite #2 on the weighted score" if (best["score"] or 0) > 0 else "; does NOT beat favourite #2 on the weighted score"))
    else:
        print("BEST none: no save has been scored yet")
    return result


if __name__ == "__main__":
    main()
