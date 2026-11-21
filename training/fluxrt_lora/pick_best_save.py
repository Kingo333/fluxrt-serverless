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
The run's score is the weighted mean over the sets (--weights; for the kiosk the studio-like sets count most).
Saves are ranked by the number of sets where the GATE passes, then by the score. Writes EVAL_DIR/best.json and
prints one SAVE line per save, an AREAS line per save (the owner's areas: colour, detail, fabric, movement,
flicker, face and skin, length), a VERDICT and a BEST line.

Chance (owner, 2026-11-17): an entry of kind "aa" is favourite #2 itself rendered with another random seed. It is
never a candidate. Its score is the size of chance differences, and the number of sets where it passes the GATE
against favourite #2 is the most any candidate is asked to pass (a model cannot be asked to be cleaner against
favourite #2 than favourite #2 is against itself). Without such an entry the bar is every set and chance is 0.

Run-time steadiness (owner, 2026-11-18): an entry of kind "runtime" is favourite #2's own file rendered with the
run-time steadiness filter (eval.steady_filter). It is never the pick either (its file is favourite #2's; the
filter is not in it): it gets a RUNTIME line instead, with its score against chance and its flicker and movement.

By garment type (owner, 2026-11-21: dresses first, but shirts and trousers too): the clip lists next to EVAL_DIR
(<set>.json, --clips) say what each test clip is (tops, dresses, bottoms). The same score is worked out again from
the per-clip numbers for the clips of one type only: a GARMENT line per save (dresses | tops | bottoms, with the
number of clips) and a GARMENT_AREAS line per type. The pick itself is unchanged: with the kiosk weights the
dresses already carry about two thirds of it.

Several chance entries (2026-11-21): with more than one "aa" entry (other random seeds) the size of chance is
the largest of theirs, per area and per garment type too (CHANCE_AREAS, GARMENT_CHANCE), and the GATE bar is the
smallest number of sets any of them passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CAP = 0.5
# metric: (lower is better, floor of the denominator: the size of a change that counts as "100%")
GARMENT = {"colour_de": (True, 1.0), "fabric": (True, 0.05), "follow_px": (True, 0.5), "jitter": (True, 0.01),
           "flicker_de": (True, 0.5), "length_err": (True, 0.03), "likeness": (False, 0.05)}
FACE = {"face_match": (False, 0.05), "skin_de": (True, 0.5), "face_drift": (True, 0.1), "texture_off": (True, 0.05)}
STEADY = {"wobble_garment": (True, 0.005), "colour_wander": (True, 0.5), "sharpness": (False, 0.05)}
DEFAULT_WEIGHTS = {"cases5": 2.0, "fx6test": 1.0, "dress10": 1.0}
# the owner's areas (2026-11-17) and the measures behind each
AREAS = (("colour", ("colour_de",)), ("detail", ("likeness",)), ("fabric", ("fabric",)), ("movement", ("follow_px",)),
         ("flicker", ("jitter", "flicker_de", "wobble_garment", "colour_wander")),
         ("face and skin", ("face_match", "skin_de", "face_drift", "texture_off")),
         ("length", ("length_err",)), ("sharpness", ("sharpness",)))


GARMENT_TYPES = ("dresses", "tops", "bottoms")
_TYPE_WORDS = {"tops": ("tops", "top", "upper", "upper_body", "upper-body", "outerwear", "shirt", "shirts", "t-shirt",
                        "blouse", "jacket", "hoodie", "sweater"),
               "bottoms": ("bottoms", "bottom", "lower", "lower_body", "lower-body", "trousers", "pants", "jeans",
                           "skirt", "skirts", "shorts")}


def garment_type(raw) -> str | None:
    """tops / dresses / bottoms from a clip's category (the app's or the dataset's word); None when unknown."""
    v = str(raw or "").strip().lower()
    if not v:
        return None
    if v.startswith("dress") or v in ("gown", "abaya", "abayas", "kaftan", "jumpsuit", "romper"):
        return "dresses"
    for t, words in _TYPE_WORDS.items():
        if v in words:
            return t
    return None


def clip_types(clips_dir: Path, sets) -> dict:
    """{set: {clip key: garment type}} from the clip lists (<clips_dir>/<set>.json); sets without a list are left out."""
    out = {}
    for s in sets:
        items = _load(Path(clips_dir) / f"{s}.json")
        if not isinstance(items, list):
            continue
        m = {str(it["key"]): t for it in items if isinstance(it, dict) and it.get("key")
             for t in [garment_type(it.get("category"))] if t}
        if m:
            out[s] = m
    return out


def _num(x):
    """A usable number, else None (a clip without that measure, or a not-a-number from an empty garment area)."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _mean(xs):
    xs = [x for x in (_num(x) for x in xs) if x is not None]
    return (sum(xs) / len(xs)) if xs else None


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


def set_gains(nums: dict) -> dict:
    """{measure: relative gain over favourite #2} for the measures that have both numbers."""
    out = {}
    for table in (GARMENT, FACE, STEADY):
        for k, (lower, floor) in table.items():
            c, b = nums.get(k, (None, None))
            x = gain(c, b, lower, floor)
            if x is not None:
                out[k] = x
    return out


def set_score(nums: dict):
    gains = list(set_gains(nums).values())
    return (sum(gains) / len(gains)) if gains else None


def area_gains(per_set_gains: dict, weights: dict) -> dict:
    """{area: weighted mean gain over the sets} (None where no set has a number for that area)."""
    out = {}
    for area, keys in AREAS:
        total = wsum = 0.0
        for s, gains in per_set_gains.items():
            vals = [gains[k] for k in keys if k in gains]
            if vals:
                total += weights.get(s, 0.0) * sum(vals) / len(vals)
                wsum += weights.get(s, 0.0)
        out[area] = round(total / wsum, 4) if wsum else None
    return out


def subset_numbers(d: Path, name: str, base: str, clips: set) -> tuple[dict, int]:
    """set_numbers for some of the set's clips only, from the per-clip records: ({metric: (cand, base)}, the number
    of those clips with garment numbers for both models)."""
    out: dict = {}
    items = (_load(d / "garment.json") or {}).get("items") or {}
    rows = [(v.get(name) or {}, v.get(base) or {}) for k, v in items.items() if k in clips and isinstance(v, dict)]
    for k in GARMENT:
        both = [(a[k], b[k]) for a, b in rows if _num(a.get(k)) is not None and _num(b.get(k)) is not None]
        if both:
            out[k] = (_mean([x for x, _ in both]), _mean([y for _, y in both]))
    n = sum(1 for a, b in rows if a and b)
    per = (_load(d / "face_skin.json") or {}).get("per_clip") or {}
    frows = [(v.get(name) or {}, v.get(base) or {}) for k, v in per.items() if k in clips and isinstance(v, dict)]
    for k in ("face_match", "skin_de", "face_drift"):
        out[k] = (_mean([a.get(k) for a, _ in frows]), _mean([b.get(k) for _, b in frows]))
    ta, tb = _mean([a.get("texture") for a, _ in frows]), _mean([b.get("texture") for _, b in frows])
    out["texture_off"] = (None if ta is None else abs(ta - 1), None if tb is None else abs(tb - 1))
    st = _load(d / "stability.json") or {}
    for k in STEADY:
        vals = {m: [clip[m].get(k) for c, clip in st.items() if c in clips and isinstance(clip, dict)
                    and isinstance(clip.get(m), dict)] for m in (name, base)}
        out[k] = (_mean(vals[name]), _mean(vals[base]))
    return out, n


def garment_split(d: Path, name: str, base: str, weights: dict, types: dict) -> dict:
    """{garment type: {"score", "clips", "areas"}} for one save: the set score again, for the clips of one type."""
    out = {}
    for t in GARMENT_TYPES:
        gains, total, wsum, n = {}, 0.0, 0.0, 0
        for s, w in weights.items():
            clips = {c for c, ct in types.get(s, {}).items() if ct == t}
            if not clips or not (d / s).is_dir():
                continue
            nums, k = subset_numbers(d / s, name, base, clips)
            g = set_gains(nums)
            if not g:
                continue
            gains[s] = g
            n += k
            total += w * sum(g.values()) / len(g)
            wsum += w
        if wsum:
            out[t] = {"score": round(total / wsum, 4), "clips": n, "areas": area_gains(gains, weights)}
    return out


def judge(root: Path, base: str, weights: dict, clips_dir: Path | None = None) -> list[dict]:
    rows = []
    types = clip_types(clips_dir if clips_dir is not None else root.parent, weights)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / ".scored").exists()):
        meta = _load(d / "meta.json") or {}
        name = meta.get("name")
        if not name:
            continue
        per_set, total, wsum, gates, face_gates, gains = {}, 0.0, 0.0, 0, 0, {}
        for s, w in weights.items():
            if not (d / s).is_dir():
                continue
            nums = set_numbers(d / s, name, base)
            sc = set_score(nums)
            gains[s] = set_gains(nums)
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
                     "kind": meta.get("kind", "save"), "label": meta.get("label") or f"step {meta.get('step')}",
                     "sha256": meta.get("sha256"), "score": round(total / wsum, 4) if wsum else None,
                     "gate_sets": gates, "face_gate_sets": face_gates, "sets": len(per_set), "per_set": per_set,
                     "areas": area_gains(gains, weights),
                     "garments": garment_split(d, name, base, weights, types)})
    return rows


def chance_size(aa: dict, weights: dict) -> float:
    """The size of chance differences in the weighted score, from favourite #2 against itself with another seed:
    the larger of its score and the weighted root-mean-square of its per-set scores (set scores of opposite sign
    must not cancel into a falsely small figure)."""
    xs = [(weights.get(s, 0.0), v["score"]) for s, v in aa["per_set"].items() if v.get("score") is not None]
    wsum = sum(w for w, _ in xs)
    rms = (sum((w * x) ** 2 for w, x in xs) ** 0.5) / wsum if wsum else 0.0
    return max(abs(aa["score"] or 0.0), rms)


def rank_key(r: dict, bar: int | None = None):
    """GATE first (no credit beyond the bar: what favourite #2 passes against itself), then the score."""
    g = r["gate_sets"] if bar is None else min(r["gate_sets"], bar)
    return (g, r["score"] if r["score"] is not None else -1e9)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--baseline", required=True, help="favourite #2's render name, e.g. fav2_r528")
    ap.add_argument("--weights", default=",".join(f"{k}={v:g}" for k, v in DEFAULT_WEIGHTS.items()))
    ap.add_argument("--clips", help="folder with the clip lists <set>.json (garment types); default: EVAL_DIR's parent")
    a = ap.parse_args(argv)
    weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    root = Path(a.eval_dir)
    rows = judge(root, a.baseline, weights, Path(a.clips) if a.clips else None)
    fmt = lambda x: "-" if x is None else f"{x * 100:+.1f}%"  # noqa: E731
    for r in sorted(rows, key=lambda r: (r["step"] if r["step"] is not None else -1)):
        sets = "; ".join(f"{s} {fmt(v['score'])}{' GATE ok' if v['gate'] else ' GATE fail'}"
                         + (f", colour {v['colour'][0]} vs {v['colour'][1]}" if v["colour"] and None not in v["colour"] else "")
                         + (f", length {v['length_err'][0]} vs {v['length_err'][1]}" if v["length_err"] and None not in v["length_err"] else "")
                         for s, v in r["per_set"].items())
        print(f"SAVE {r['label']} ({r['name']}): score {fmt(r['score'])} vs favourite #2; GATE {r['gate_sets']}/{r['sets']} sets; {sets}")
        areas = ", ".join(f"{k} {fmt(v)}" for k, v in r["areas"].items() if v is not None)
        if areas:
            print(f"AREAS {r['label']}: {areas}")
        if r["garments"]:
            print(f"GARMENT {r['label']}: "
                  + " | ".join(f"{t} {fmt(g['score'])} ({g['clips']} clips)" for t, g in r["garments"].items()))
            for t, g in r["garments"].items():
                ga = ", ".join(f"{k} {fmt(v)}" for k, v in g["areas"].items() if v is not None)
                if ga:
                    print(f"GARMENT_AREAS {r['label']}, {t}: {ga}")
    # chance: favourite #2 against itself with other random seeds. The first one is "the" chance entry; with
    # several, every size is the largest of theirs and the GATE bar the smallest number of sets one of them passes.
    aas = sorted((r for r in rows if r["kind"] == "aa"), key=lambda r: -(r["step"] if r["step"] is not None else 0))
    aa = aas[0] if aas else None
    runtime = sorted((r for r in rows if r["kind"] == "runtime"), key=lambda r: (r["step"] if r["step"] is not None else 0))
    cands = [r for r in rows if r["kind"] not in ("aa", "runtime")]
    bar = min(x["gate_sets"] for x in aas) if aas else None
    noise = max(chance_size(x, weights) for x in aas) if aas else 0.0
    area_noise = {k: max(abs(x["areas"][k]) for x in aas if x["areas"].get(k) is not None)
                  for k, _ in AREAS if any(x["areas"].get(k) is not None for x in aas)}
    garment_noise = {t: max(abs(x["garments"][t]["score"]) for x in aas if t in x["garments"])
                     for t in GARMENT_TYPES if any(t in x["garments"] for x in aas)}
    best = max(cands, key=lambda r: rank_key(r, bar)) if cands else None
    result = {"baseline": a.baseline, "weights": weights, "saves": rows, "best": best,
              "chance": None if aa is None else {"score": aa["score"], "size": round(noise, 4),
                                                 "gate_sets": bar, "sets": aa["sets"], "seeds": len(aas),
                                                 "areas": {k: round(v, 4) for k, v in area_noise.items()},
                                                 "garments": {t: round(v, 4) for t, v in garment_noise.items()}},
              "runtime": [{"label": r["label"], "name": r["name"], "score": r["score"], "areas": r["areas"],
                           "beyond_chance": (r["score"] or 0) > noise} for r in runtime]}
    (root / "best.json").write_text(json.dumps(result, indent=1))
    size = lambda x: f"{abs(x) * 100:.1f}%"  # noqa: E731
    if len(aas) == 1:
        print(f"CHANCE favourite #2 with another random seed: score {fmt(aa['score'])} against itself, "
              f"GATE {aa['gate_sets']}/{aa['sets']} sets; differences up to {fmt(noise)} mean nothing")
    elif aas:
        print(f"CHANCE favourite #2 with {len(aas)} other random seeds: scores "
              + ", ".join(fmt(x["score"]) for x in aas) + " against itself, GATE "
              + ", ".join(f"{x['gate_sets']}/{x['sets']}" for x in aas)
              + f" sets; differences up to {fmt(noise)} mean nothing")
    if area_noise:
        print("CHANCE_AREAS differences up to these sizes mean nothing: "
              + ", ".join(f"{k} {size(v)}" for k, v in area_noise.items()))
    if garment_noise:
        print("GARMENT_CHANCE differences up to these sizes mean nothing: "
              + " | ".join(f"{t} {size(v)}" for t, v in garment_noise.items()))
    for r in runtime:
        ar = r["areas"]
        beyond = (r["score"] or 0) > noise
        print(f"RUNTIME {r['label']}: score {fmt(r['score'])} vs favourite #2 ({'beyond' if beyond else 'within'} chance "
              f"{fmt(noise)}); flicker {fmt(ar.get('flicker'))}, movement {fmt(ar.get('movement'))}, detail "
              f"{fmt(ar.get('detail'))}, colour {fmt(ar.get('colour'))}: "
              + ("worth a live test (watch its video for smearing on moving arms)" if beyond else "no clear gain"))
    if best:
        need = best["sets"] if bar is None else min(bar, best["sets"])
        full = best["gate_sets"] >= need and (best["score"] or 0) > noise
        if best["garments"] and garment_noise:
            def word(t, g):
                n = garment_noise.get(t)
                if n is None:
                    return f"{t} {fmt(g['score'])}"
                return f"{t} {fmt(g['score'])} ({'better' if g['score'] > n else 'worse' if g['score'] < -n else 'level'}, chance {size(n)})"
            print(f"GARMENT_VERDICT {best['label']} vs favourite #2: "
                  + " | ".join(word(t, g) for t, g in best["garments"].items()))
        if aa is None:
            print("VERDICT " + (f"{best['label']} beats favourite #2 (GATE on every set and a better score): a candidate for live"
                                if full else
                                f"nothing beats favourite #2 on every set: keep favourite #2 live (closest: {best['label']}, "
                                f"score {fmt(best['score'])}, GATE {best['gate_sets']}/{best['sets']} sets)"))
        else:
            print("VERDICT " + (f"{best['label']} beats favourite #2 beyond chance (score {fmt(best['score'])}, chance "
                                f"{fmt(noise)}; GATE {best['gate_sets']}/{best['sets']} sets, favourite #2 against itself "
                                f"{bar}/{aa['sets']}): a candidate for a kiosk test"
                                if full else
                                f"nothing beats favourite #2 beyond chance: keep favourite #2 live (closest: {best['label']}, "
                                f"score {fmt(best['score'])}, chance {fmt(noise)}, GATE {best['gate_sets']}/{best['sets']} sets, "
                                f"favourite #2 against itself {bar}/{aa['sets']})"))
        print(f"BEST {best['label']} ({best['name']}): score {fmt(best['score'])}, GATE {best['gate_sets']}/{best['sets']} sets"
              + ("; beats favourite #2 on the weighted score" if (best["score"] or 0) > 0 else "; does NOT beat favourite #2 on the weighted score"))
    else:
        print("BEST none: no save has been scored yet")
    return result


if __name__ == "__main__":
    main()
