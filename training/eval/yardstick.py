"""The better yardstick (owner, 2026-11-21): is a model, or a random seed, really different from favourite #2?

Why: favourite #2 rendered with another random seed scored up to +4.2% against itself (flicker +8.5%), so one render
of 39 clips cannot show a difference smaller than about 4%. This tool compares on MORE clips (a fresh set next to
the old ones), at SEVERAL seeds, and gives every figure a 95% range from the per-clip numbers (clips re-drawn with
replacement, 1000 times, inside each set: the range says what another draw of test clips could have shown).

  python -m eval.yardstick YARD_DIR --baseline fav2_r528 --old fx6test dress10 cases5 --fresh fx12test \
      [--weights fx6test=1,dress10=1,cases5=0.5,fx12test=2] [--clips DIR] [--boot 1000] [--json OUT]

YARD_DIR/<entry>/meta.json: {"name": render name, "label", "model": "fav2" or a model's label, "seed": N}, and per
set <set>/garment.json, face_skin.json, stability.json as the scorers write them (each holds the entry's own
per-clip numbers and favourite #2's at the live seed).

The score is the picker's (fluxrt_lora.pick_best_save): per set, every measure's relative gain over the other side
(+ = better, capped), averaged; the sets weighted. Lines:
  YARD_SETS         the clips: old and fresh, per garment type
  YARD_SEED         favourite #2 at another seed against the live seed: on the old clips (where the seed was
                    picked) and on the fresh clips (where it was not): only the fresh figure is a fair test
  YARD_SEED_ORDER   the seeds best first, on the old and on the fresh clips: the same order = seeds really differ
  YARD_MODEL        a model against favourite #2 AT THE SAME SEEDS, all clips: the seeds pooled, and each seed alone
  YARD_MODEL_GARMENT / YARD_MODEL_AREAS   the same per garment type and per area
  YARD_RESOLUTION   how small a difference this yardstick shows
A range that does not include 0 is a real difference ("better" or "worse"); otherwise "no difference shown".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fluxrt_lora.pick_best_save import AREAS, CAP, FACE, GARMENT, GARMENT_TYPES, STEADY, clip_types

# every measure: (key in the records, lower is better, floor); "texture" is turned into |texture - 1| on the mean
MEASURES = ([(k, *GARMENT[k]) for k in GARMENT] + [(k, *FACE[k]) for k in ("face_match", "skin_de", "face_drift")]
            + [("texture", *FACE["texture_off"])] + [(k, *STEADY[k]) for k in STEADY])
KEYS = [k for k, _, _ in MEASURES]
LOWER = np.array([lo for _, lo, _ in MEASURES])
FLOOR = np.array([fl for _, _, fl in MEASURES])
TEX = KEYS.index("texture")
AREA_IDX = {a: [KEYS.index("texture" if k == "texture_off" else k) for k in ks] for a, ks in AREAS}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) else np.nan


def clip_rows(d: Path, name: str) -> dict:
    """{clip: the 14 numbers of model `name`} from one entry's records of one set (nan where there is none)."""
    out: dict = {}
    def row(clip):
        return out.setdefault(clip, np.full(len(KEYS), np.nan))
    for clip, v in ((_load(d / "garment.json") or {}).get("items") or {}).items():
        m = v.get(name) if isinstance(v, dict) else None
        if isinstance(m, dict):
            for k in GARMENT:
                row(clip)[KEYS.index(k)] = _num(m.get(k))
    for clip, v in ((_load(d / "face_skin.json") or {}).get("per_clip") or {}).items():
        m = v.get(name) if isinstance(v, dict) else None
        if isinstance(m, dict):
            for k in ("face_match", "skin_de", "face_drift", "texture"):
                row(clip)[KEYS.index(k)] = _num(m.get(k))
    for clip, v in (_load(d / "stability.json") or {}).items():
        m = v.get(name) if isinstance(v, dict) else None
        if isinstance(m, dict):
            for k in STEADY:
                row(clip)[KEYS.index(k)] = _num(m.get(k))
    return out


def load(yard: Path, baseline: str, sets: list[str], live: int = 52) -> dict:
    """{(model, seed): {set: {clip: numbers}}}. Favourite #2 at the live seed is ("fav2", live): it has no entry of
    its own, its numbers come with every other entry's records (every scoring holds both models)."""
    table: dict = {}
    for d in sorted(p for p in yard.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        meta = _load(d / "meta.json") or {}
        if not meta.get("name") or meta.get("seed") is None or not meta.get("model"):
            continue
        key = (str(meta["model"]), int(meta["seed"]))
        if key == ("fav2", live):
            continue
        for s in sets:
            if not (d / s).is_dir():
                continue
            rows = clip_rows(d / s, meta["name"])
            if rows:
                table.setdefault(key, {})[s] = rows
            dst = table.setdefault(("fav2", live), {}).setdefault(s, {})
            for c, r in clip_rows(d / s, baseline).items():
                dst.setdefault(c, r)
    return table


def _gains(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-measure relative gain of the means c over b (nan where a side is missing), as the picker counts it."""
    c, b = c.copy(), b.copy()
    c[TEX], b[TEX] = abs(c[TEX] - 1), abs(b[TEX] - 1)
    d = np.where(LOWER, b - c, c - b)
    return np.clip(d / np.maximum(np.abs(b), FLOOR), -CAP, CAP)


def _mean(x: np.ndarray) -> np.ndarray:
    """Column means that skip nan, nan for a column without any number (no warning)."""
    n = np.sum(~np.isnan(x), axis=0)
    return np.where(n > 0, np.nansum(x, axis=0) / np.maximum(n, 1), np.nan)


class Pairs:
    """One side against the other on the same clips (and the same seeds): per set, the paired numbers."""

    def __init__(self, table: dict, a: str, b: str, seeds: list[tuple[int, int]], sets: list[str], weights: dict,
                 only: dict | None = None, kind: str | None = None):
        # seeds: [(seed of side a, seed of side b)]; only: {set: {clip: garment type}} with kind = the type wanted
        self.sets, self.weights = [], []
        self.a, self.b = [], []                      # per set: arrays (clips, seeds, measures)
        for s in sets:
            clips = None
            for sa, sb in seeds:
                ra, rb = table.get((a, sa), {}).get(s), table.get((b, sb), {}).get(s)
                both = set(ra or {}) & set(rb or {})
                clips = both if clips is None else clips & both
            if kind is not None:
                clips = {c for c in (clips or set()) if (only or {}).get(s, {}).get(c) == kind}
            if not clips:
                continue
            order = sorted(clips)
            A = np.array([[table[(a, sa)][s][c] for sa, _ in seeds] for c in order])
            B = np.array([[table[(b, sb)][s][c] for _, sb in seeds] for c in order])
            miss = np.isnan(A) | np.isnan(B)         # a measure counts only where both sides have it
            A[miss], B[miss] = np.nan, np.nan
            self.sets.append(s)
            self.weights.append(weights.get(s, 1.0))
            self.a.append(A)
            self.b.append(B)
        self.clips = sum(len(x) for x in self.a)

    def _score(self, pick=None, seed_col=None):
        """(score, {area: gain}) over the sets; pick = per set the clip indices of one re-draw."""
        tot = wsum = 0.0
        area_tot = {k: 0.0 for k in AREA_IDX}
        area_w = {k: 0.0 for k in AREA_IDX}
        for i, w in enumerate(self.weights):
            A, B = self.a[i], self.b[i]
            if pick is not None:
                A, B = A[pick[i]], B[pick[i]]
            if seed_col is not None:
                A, B = A[:, seed_col:seed_col + 1], B[:, seed_col:seed_col + 1]
            g = _gains(_mean(A.reshape(-1, len(KEYS))), _mean(B.reshape(-1, len(KEYS))))
            if np.all(np.isnan(g)) or w <= 0:
                continue
            tot += w * float(np.nanmean(g))
            wsum += w
            for k, idx in AREA_IDX.items():
                v = g[idx]
                if not np.all(np.isnan(v)):
                    area_tot[k] += w * float(np.nanmean(v))
                    area_w[k] += w
        if not wsum:
            return None, {}
        return tot / wsum, {k: area_tot[k] / area_w[k] for k in AREA_IDX if area_w[k]}

    def result(self, boot: int = 1000, rng_seed: int = 7) -> dict | None:
        score, areas = self._score()
        if score is None:
            return None
        rng = np.random.default_rng(rng_seed)
        draws, area_draws = [], {k: [] for k in areas}
        for _ in range(boot):
            pick = [rng.integers(0, len(x), len(x)) for x in self.a]
            sc, ar = self._score(pick)
            if sc is not None:
                draws.append(sc)
                for k in area_draws:
                    if k in ar:
                        area_draws[k].append(ar[k])
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
        n_seeds = self.a[0].shape[1] if self.a else 0
        return {"score": score, "low": float(lo), "high": float(hi), "clips": self.clips, "sets": list(self.sets),
                "areas": {k: {"score": v, "low": float(np.percentile(area_draws[k], 2.5)),
                              "high": float(np.percentile(area_draws[k], 97.5))} for k, v in areas.items() if area_draws[k]},
                "per_seed": [self._score(seed_col=j)[0] for j in range(n_seeds)]}


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.1f}%"


def rng_txt(r: dict) -> str:
    return f"{pct(r['score'])} (95% {pct(r['low'])} to {pct(r['high'])})"


def word(r: dict) -> str:
    return "better" if r["low"] > 0 else "worse" if r["high"] < 0 else "no difference shown"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yard")
    ap.add_argument("--baseline", required=True, help="favourite #2's render name at the live seed, e.g. fav2_r528")
    ap.add_argument("--old", nargs="+", required=True, help="the sets the seeds were picked on")
    ap.add_argument("--fresh", nargs="*", default=[], help="sets never used before")
    ap.add_argument("--weights", default="")
    ap.add_argument("--clips", help="folder with the clip lists <set>.json; default: YARD's parent")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--live-seed", type=int, default=52, help="the seed of the live worker and of the usual renders")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    yard = Path(a.yard)
    sets = list(a.old) + list(a.fresh)
    weights = {s: 1.0 for s in sets}
    weights.update({k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)})
    types = clip_types(Path(a.clips) if a.clips else yard.parent, sets)
    live = a.live_seed
    table = load(yard, a.baseline, sets, live)
    out: dict = {"sets": {}, "seeds": [], "models": []}
    base = table.get(("fav2", live), {})
    for label, group in (("old", a.old), ("fresh", a.fresh)):
        if not group:
            continue
        n = {t: sum(1 for s in group for c in base.get(s, {}) if types.get(s, {}).get(c) == t) for t in GARMENT_TYPES}
        total = sum(len(base.get(s, {})) for s in group)
        out["sets"][label] = {"sets": group, "clips": total, **n}
        print(f"YARD_SETS {label} clips: {total} in {', '.join(group)} ("
              + ", ".join(f"{n[t]} {t}" for t in GARMENT_TYPES) + ")")
    half = []
    # ---- favourite #2 at other seeds against the live seed
    for seed in sorted(s for (m, s) in table if m == "fav2" and s != live):
        ro = Pairs(table, "fav2", "fav2", [(seed, live)], a.old, weights).result(a.boot)
        rf = Pairs(table, "fav2", "fav2", [(seed, live)], a.fresh, weights).result(a.boot) if a.fresh else None
        if ro is None and rf is None:
            continue
        txt = f"YARD_SEED favourite #2 at seed {seed} against seed {live}:"
        if ro:
            txt += f" old clips {rng_txt(ro)}"
        if rf:
            txt += f"{';' if ro else ''} fresh clips {rng_txt(rf)}: " + \
                   {"better": "holds on fresh clips", "worse": "worse on fresh clips"}.get(word(rf), "not shown on fresh clips")
            half.append((rf["high"] - rf["low"]) / 2)
            ar = ", ".join(f"{k} {rng_txt(v)}" for k, v in rf["areas"].items() if k in ("flicker", "colour", "fabric", "detail"))
            if ar:
                print(txt)
                txt = f"YARD_SEED_AREAS seed {seed}, fresh clips: {ar}"
        print(txt)
        out["seeds"].append({"seed": seed, "old": ro, "fresh": rf})
    def order(which):
        xs = sorted(((x[which]["score"], x["seed"]) for x in out["seeds"] if x.get(which)), reverse=True)
        return " > ".join(str(sd) for _, sd in xs)
    if len(out["seeds"]) >= 2:
        print(f"YARD_SEED_ORDER best first, on the old clips: {order('old')}"
              + (f"; on the fresh clips: {order('fresh')}" if a.fresh and any(x.get('fresh') for x in out['seeds']) else ""))
    # ---- every other model against favourite #2 at the same seeds, all clips
    for model in sorted({m for (m, _) in table if m != "fav2"}):
        seeds = sorted(s for (m, s) in table if m == model and ("fav2", s) in table)
        if not seeds:
            continue
        pairs = [(s, s) for s in seeds]
        r = Pairs(table, model, "fav2", pairs, sets, weights).result(a.boot)
        if r is None:
            continue
        half.append((r["high"] - r["low"]) / 2)
        print(f"YARD_MODEL {model} against favourite #2 at the same seeds ({', '.join(map(str, seeds))}), {r['clips']} clips: "
              f"{rng_txt(r)}: {word(r)}; each seed alone " + ", ".join(pct(x) for x in r["per_seed"]))
        g = {}
        for t in GARMENT_TYPES:
            rt = Pairs(table, model, "fav2", pairs, sets, weights, only=types, kind=t).result(a.boot)
            if rt:
                g[t] = rt
        if g:
            print(f"YARD_MODEL_GARMENT {model}: " + " | ".join(f"{t} {rng_txt(v)} {word(v)}, {v['clips']} clips" for t, v in g.items()))
        print(f"YARD_MODEL_AREAS {model}: " + ", ".join(f"{k} {rng_txt(v)}" for k, v in r["areas"].items()))
        out["models"].append({"model": model, "seeds": seeds, "all": r, "garments": g})
    if half:
        res = float(np.median(half))
        out["resolution"] = res
        print(f"YARD_RESOLUTION this yardstick shows differences of about {res * 100:.1f}% or more "
              f"(half the width of a typical 95% range)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
