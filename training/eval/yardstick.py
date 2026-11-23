"""The better yardstick (owner, 2026-11-21): is a model, or a random seed, really different from favourite #2?

Why: favourite #2 rendered with another random seed scored up to +4.2% against itself (flicker +8.5%), so one render
of 39 clips cannot show a difference smaller than about 4%. This tool compares on MORE clips (a fresh set next to
the old ones), at SEVERAL seeds, and gives every figure a 95% range from the per-clip numbers (clips re-drawn with
replacement, 1000 times, inside each set: the range says what another draw of test clips could have shown).

  python -m eval.yardstick YARD_DIR --baseline fav2_r528 --old fx6test dress10 cases5 --fresh fx12test \
      [--weights fx6test=1,dress10=1,cases5=0.5,fx12test=2] [--clips DIR] [--boot 1000] [--worn DIR] [--json OUT]

YARD_DIR/<entry>/meta.json: {"name": render name, "label", "model": "fav2" or a model's label, "seed": N}, and per
set <set>/garment.json, face_skin.json, stability.json as the scorers write them (each holds the entry's own
per-clip numbers and favourite #2's at the live seed).

The score is the picker's (fluxrt_lora.pick_best_save): per set, every measure's relative gain over the other side
(+ = better, capped), averaged; the sets weighted. Lines:
  YARD_SETS         the clips: old and fresh, per garment type
  YARD_SEED         favourite #2 at another seed against the live seed: on the old clips (where the seed was
                    picked) and on the fresh clips (where it was not): only the fresh figure is a fair test
  YARD_SEED_ORDER   the seeds best first, on the old and on the fresh clips: the same order = seeds really differ
  YARD_SEED_AVERAGE all the other seeds together against the live seed: is the live seed a poor one? With a
                    YARD_SEED_AVERAGE_GARMENT line per garment type
  YARD_SEED_AGREEMENT  how well the seeds' order on the old clips predicts their order on the fresh clips
                    (rank correlation: 1 = the same order, 0 = no relation)
  YARD_SEED_PICK    the seeds that are better than the live seed on the fresh clips at the level that allows for
                    the number of seeds tried (with 12 seeds one passes the ordinary 95% test by luck in about one
                    run in four), with no area clearly worse on BOTH the old and the fresh clips (worse on one of
                    them only is a remark beside the seed); the seeds that fail one of the two are named apart
                    with the reason (YARD_SEED_GARMENT: the best three per garment type)
  YARD_CONTROL      favourite #2 rendered AGAIN at the live seed against its stored videos: must be 0. On the fresh
                    clips both renders are today's (how repeatable a render is); on the old clips it shows whether
                    the stored videos still are what today's renderer makes. YARD_CONTROL_SEEDS: the other seeds
                    against that fresh render on the old clips (the fair figure if the stored videos differ)
  YARD_MODEL        a model against favourite #2 AT THE SAME SEEDS, all clips: the seeds pooled, each seed alone, and
                    the fresh clips alone (all their videos are from one run)
  YARD_MODEL_GARMENT / YARD_MODEL_AREAS   the same per garment type and per area
  YARD_FABRIC       the direction of the fabric score (it has none itself): the garment's light-dark spread as a
                    multiple of the photo's, for favourite #2 per seed and for each model next to favourite #2 at
                    the same seeds and on the same clips (under 1 = flatter than the photo)
  YARD_REAL         --worn: the same garments as REALLY worn (eval.worn_reference: the real videos of the garment
                    photos' own clips, scored like a render) next to favourite #2 on the same garments, measure by
                    measure: the level a perfect try-on would reach. YARD_REAL_GARMENT: per garment type
  YARD_GAPS         --worn: per model at the live seed, where favourite #2 stands apart from the real videos in a
                    way these measures can see (colour flicker on tops, light-dark spread per garment type): the
                    model's figure next to favourite #2's and the real videos', and whether it is nearer to real.
                    A garment's length and shape are not here: they are judged by eye in the videos
  YARD_RESOLUTION   how small a difference this yardstick shows
  YARD_NOTE         score files that are there but cannot be read (left out, never taken for scores)
A range that does not include 0 is a real difference ("better" or "worse"); otherwise "no difference shown"; with
fewer than 5 clips nothing is called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fluxrt_lora.pick_best_save import AREAS, CAP, FACE, GARMENT, GARMENT_TYPES, STEADY, clip_types

CONTROL = "favourite #2 rendered again"      # the model name of the same-seed control (fxb_yard.sh)
WORN = "worn"                                # the render name of the real videos (eval.worn_reference)
# what the real videos are shown with: (key, words, format, a ratio: averaged as a ratio)
SHOWN = (("colour_de", "colour distance from the photo", "{:.1f}", False), ("likeness", "likeness to the photo", "{:.2f}", False),
         ("fabric_ratio", "light-dark spread (1 = the photo's)", "x{:.2f}", True),
         ("length_ratio", "dress length over waist width (1 = the photo's)", "x{:.2f}", True),
         ("flicker_de", "colour flicker", "{:.2f}", False), ("jitter", "shape jitter", "{:.3f}", False),
         ("follow_px", "garment off the body's movement", "{:.2f} px", False), ("wobble_garment", "wobble", "{:.2f}", False),
         ("colour_wander", "colour wander", "{:.2f}", False), ("sharpness", "sharpness", "{:.1f}", False))

# every measure: (key in the records, lower is better, floor); "texture" is turned into |texture - 1| on the mean
MEASURES = ([(k, *GARMENT[k]) for k in GARMENT] + [(k, *FACE[k]) for k in ("face_match", "skin_de", "face_drift")]
            + [("texture", *FACE["texture_off"])] + [(k, *STEADY[k]) for k in STEADY])
KEYS = [k for k, _, _ in MEASURES]
LOWER = np.array([lo for _, lo, _ in MEASURES])
FLOOR = np.array([fl for _, _, fl in MEASURES])
TEX = KEYS.index("texture")
AREA_IDX = {a: [KEYS.index("texture" if k == "texture_off" else k) for k in ks] for a, ks in AREAS}


UNREADABLE: list = []        # score files that are there but cannot be read (a Pod stopped while one was written)


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except OSError:
        return None
    except ValueError:
        if str(p) not in UNREADABLE:
            UNREADABLE.append(str(p))
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


def fabric_ratios(d: Path, name: str) -> dict:
    """{clip: the garment's light-dark spread over the photo's} for model `name` (newer scorings only)."""
    out = {}
    for clip, v in ((_load(d / "garment.json") or {}).get("items") or {}).items():
        m = v.get(name) if isinstance(v, dict) else None
        x = _num(m.get("fabric_ratio")) if isinstance(m, dict) else np.nan
        if x > 0:
            out[clip] = x
    return out


def raw_values(d: Path, name: str) -> dict:
    """{clip: {key: number}} of model `name` in one scored set: the SHOWN keys as the scorers wrote them."""
    keys = {k for k, _, _, _ in SHOWN}
    out: dict = {}
    per = [((_load(d / "garment.json") or {}).get("items") or {}), (_load(d / "stability.json") or {})]
    for src in per:
        for clip, v in src.items():
            m = v.get(name) if isinstance(v, dict) else None
            if isinstance(m, dict):
                for k in keys & set(m):
                    x = _num(m[k])
                    if not np.isnan(x):
                        out.setdefault(clip, {})[k] = x
    return out


def load_raw(yard: Path, baseline: str, sets: list[str]) -> dict:
    """{(set, clip): {key: number}} of favourite #2 at the live seed (every scoring in the yard holds it)."""
    out: dict = {}
    for d in sorted(p for p in yard.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        for s in sets:
            if (d / s).is_dir():
                for c, row in raw_values(d / s, baseline).items():
                    dst = out.setdefault((s, c), {})
                    for k, x in row.items():
                        dst.setdefault(k, x)
    return out


def real_against(real: dict, fav: dict, clips=None) -> tuple[str, dict, int]:
    """The real videos next to favourite #2 on the same garments: (the words, {key: {real, fav2, clips}}, clips)."""
    both = [c for c in real if c in fav and (clips is None or c in clips)]
    parts, res = [], {}
    for k, label, fmt, ratio in SHOWN:
        rows = [(real[c][k], fav[c][k]) for c in both if k in real[c] and k in fav[c] and (not ratio or (real[c][k] > 0 and fav[c][k] > 0))]
        if len(rows) < 3:
            continue
        agg = geo if ratio else (lambda xs: float(np.mean(list(xs))))
        r, f = agg([x for x, _ in rows]), agg([y for _, y in rows])
        res[k] = {"real": r, "fav2": f, "clips": len(rows)}
        parts.append(f"{label}: real {fmt.format(r)}, favourite #2 {fmt.format(f)}")
    return " | ".join(parts), res, len(both)


# where favourite #2 stands apart from the real videos (yardstick run, 2026-11-22): key, words, garment types.
# The dress length is NOT among them: length_ratio is the dress's length over its narrowest waist row, so a dress
# that comes out straighter, or with a hand or a bag in front of the waist, reads as "longer". The owner's review
# of the videos (2026-11-23) showed dresses coming out too SHORT where this figure said 15% longer. Length and
# shape are judged by eye, in the side-by-side videos.
GAPS = (("flicker_de", "colour flicker on tops", ("tops",)), ("fabric_ratio", "light-dark spread", GARMENT_TYPES))


def entries(yard: Path) -> list[tuple[Path, dict]]:
    """Every entry of the yard that says what it is: (folder, meta)."""
    out = []
    for d in sorted(p for p in yard.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        meta = _load(d / "meta.json") or {}
        if meta.get("name") and meta.get("seed") is not None and meta.get("model"):
            out.append((d, meta))
    return out


def load_model_raw(yard: Path, sets: list[str], live: int = 52) -> dict:
    """{model: {(set, clip): {key: number}}} of every model (not favourite #2, not the control) at the live seed."""
    out: dict = {}
    for d, meta in entries(yard):
        if str(meta["model"]) in ("fav2", CONTROL) or int(meta["seed"]) != live:
            continue
        for st in sets:
            if (d / st).is_dir():
                for c, row in raw_values(d / st, meta["name"]).items():
                    out.setdefault(str(meta["model"]), {})[(st, c)] = row
    return out


def gaps_against(real: dict, fav: dict, mine: dict, types: dict) -> tuple[str, dict]:
    """One model next to favourite #2 and the real videos on the same garments, where favourite #2 stands apart
    from real: (the words, {key:type: {model, fav2, real, clips, word}})."""
    ratio = {k: r for k, _, _, r in SHOWN}
    fmt = {k: f for k, _, f, _ in SHOWN}
    parts, res = [], {}
    for key, words, kinds in GAPS:
        for kind in kinds:
            cl = [c for c in real if c in fav and c in mine and types.get(c[0], {}).get(c[1]) == kind
                  and all(x[c].get(key, 0) > 0 for x in (real, fav, mine))]
            if len(cl) < 3:
                continue
            agg = geo if ratio[key] else (lambda xs: float(np.mean(list(xs))))
            r, f, m = (agg([x[c][key] for c in cl]) for x in (real, fav, mine))
            dm, df = abs(float(np.log(m / r))), abs(float(np.log(f / r)))
            word_ = "nearer to real" if dm < df - 0.02 else "further from real" if dm > df + 0.02 else "as near to real"
            name = words if len(kinds) == 1 else f"{words} on {kind}"
            parts.append(f"{name} {fmt[key].format(m)} (favourite #2 {fmt[key].format(f)}, real {fmt[key].format(r)}): {word_}")
            res[f"{key}:{kind}"] = {"model": m, "fav2": f, "real": r, "clips": len(cl), "word": word_}
    return " | ".join(parts), res


def load_fabric(yard: Path, baseline: str, sets: list[str], live: int = 52) -> dict:
    """{(model, seed): {(set, clip): ratio}}, favourite #2 at the live seed included (as in load)."""
    table: dict = {}
    for d in sorted(p for p in yard.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        meta = _load(d / "meta.json") or {}
        if not meta.get("name") or meta.get("seed") is None or not meta.get("model"):
            continue
        key = (str(meta["model"]), int(meta["seed"]))
        for s in sets:
            if not (d / s).is_dir():
                continue
            if key != ("fav2", live):
                for c, x in fabric_ratios(d / s, meta["name"]).items():
                    table.setdefault(key, {})[(s, c)] = x
            for c, x in fabric_ratios(d / s, baseline).items():
                table.setdefault(("fav2", live), {}).setdefault((s, c), x)
    return table


def geo(xs) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


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

    def result(self, boot: int = 1000, rng_seed: int = 7, strict: float | None = None) -> dict | None:
        """strict: a lower percentile (e.g. 2.5 / the number of things tried) reported as low_strict: what the range's
        lower end would be if this were one of several tries and no try may pass by luck."""
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
        extra = {"low_strict": float(np.percentile(draws, strict))} if strict is not None and draws else {}
        return {"score": score, "low": float(lo), "high": float(hi), **extra, "clips": self.clips, "sets": list(self.sets),
                "areas": {k: {"score": v, "low": float(np.percentile(area_draws[k], 2.5)),
                              "high": float(np.percentile(area_draws[k], 97.5))} for k, v in areas.items() if area_draws[k]},
                "per_seed": [self._score(seed_col=j)[0] for j in range(n_seeds)]}


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.1f}%"


def rng_txt(r: dict) -> str:
    return f"{pct(r['score'])} (95% {pct(r['low'])} to {pct(r['high'])})"


MIN_CLIPS = 5        # under this many clips a range from re-drawn clips says nothing: no "better" or "worse"
FLAW = 0.02          # an area is "clearly worse" when its range is under 0 and it is worse by more than this


def word(r: dict) -> str:
    if r.get("clips", MIN_CLIPS) < MIN_CLIPS:
        return "too few clips to say"
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
    ap.add_argument("--worn", help="the scores of the real videos of the test garments: WORN/<set>/garment.json, stability.json")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    yard = Path(a.yard)
    UNREADABLE.clear()
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
    tried = sorted(sd for (m, sd) in table if m == "fav2" and sd != live and any(st in table[(m, sd)] for st in a.fresh))
    n_tried = max(1, len(tried))
    boot_seed = max(a.boot, min(400 * n_tried, 5 * a.boot))     # enough re-draws for the strict lower end
    for seed in sorted(s for (m, s) in table if m == "fav2" and s != live):
        ro = Pairs(table, "fav2", "fav2", [(seed, live)], a.old, weights).result(a.boot)
        rf = Pairs(table, "fav2", "fav2", [(seed, live)], a.fresh, weights).result(boot_seed, strict=2.5 / n_tried) if a.fresh else None
        if ro is None and rf is None:
            continue
        txt = f"YARD_SEED favourite #2 at seed {seed} against seed {live}:"
        if ro:
            txt += f" old clips {rng_txt(ro)}"
        if rf:
            txt += f"{';' if ro else ''} fresh clips {rng_txt(rf)}: " + \
                   {"better": "holds on fresh clips", "worse": "worse on fresh clips"}.get(word(rf), "not shown on fresh clips")
            half.append((rf["high"] - rf["low"]) / 2)
            ar = ", ".join(f"{k} {rng_txt(v)}" for k, v in rf["areas"].items())
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
    # ---- all the other seeds together: is the live seed a poor one?
    others = sorted(sd for (m, sd) in table if m == "fav2" and sd != live)

    def seeds_on(group):
        return [sd for sd in others if any(st in table.get(("fav2", sd), {}) for st in group)]

    avg = {}
    for label, group in (("old", a.old), ("fresh", a.fresh)):
        sds = seeds_on(group) if group else []
        r = Pairs(table, "fav2", "fav2", [(sd, live) for sd in sds], group, weights).result(a.boot) if len(sds) >= 2 else None
        if r:
            avg[label] = {**r, "seeds": sds}
    if avg:
        out["seed_average"] = avg
        last = avg.get("fresh") or avg["old"]
        where = "fresh" if "fresh" in avg else "old"
        print(f"YARD_SEED_AVERAGE the {len(last['seeds'])} other seeds together against seed {live}: "
              + "; ".join(f"{label} clips {rng_txt(r)}" for label, r in avg.items()) + ": "
              + {"better": f"seed {live} is a poor seed on the {where} clips"
                           + (" too" if where == "fresh" and "old" in avg and word(avg["old"]) == "better" else ""),
                 "worse": f"seed {live} is a good seed on the {where} clips"}.get(
                  word(last), f"on the {where} clips seed {live} is not shown to differ from the others"))
        group = a.fresh if "fresh" in avg else a.old
        g = {}
        for t in GARMENT_TYPES:
            rt = Pairs(table, "fav2", "fav2", [(sd, live) for sd in last["seeds"]], group, weights, only=types, kind=t).result(a.boot)
            if rt:
                g[t] = rt
        if g:
            avg["garments"] = g
            print(f"YARD_SEED_AVERAGE_GARMENT the other seeds together, {where} clips: "
                  + " | ".join(f"{t} {rng_txt(v)} {word(v)}, {v['clips']} clips" for t, v in g.items()))
    both = [(x["old"]["score"], x["fresh"]["score"]) for x in out["seeds"] if x.get("old") and x.get("fresh")]
    if len(both) >= 4:
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)                # noqa: E731
        rho = float(np.corrcoef(rk([b[0] for b in both]), rk([b[1] for b in both]))[0, 1])
        if np.isfinite(rho):
            out["seed_agreement"] = {"rank_correlation": rho, "seeds": len(both)}
            print(f"YARD_SEED_AGREEMENT the order of the {len(both)} seeds on the old clips against their order on the "
                  f"fresh clips: rank correlation {rho:+.2f} (1 = the same order, 0 = no relation): "
                  + ("a seed that is good on some clips is good on others" if rho >= 0.6 else
                     "some relation" if rho >= 0.3 else "which seed looks best depends on the clips: mostly luck"))
    # ---- which seed, if any. Several seeds are tried, so one can pass the ordinary 95% test by luck (12 seeds: about
    # one run in four): a pick must stand at the level that allows for all the seeds tried (the range's lower end
    # at 2.5% / the number of seeds). An area counts against a seed only when it is shown worse on BOTH the old and
    # the fresh clips (with 8 areas, 2 groups of clips and 12 seeds, an area is "worse" on one group by luck more
    # than once a run); worse on one group only is said beside the seed, as a remark.
    def worse_areas(x, lab):
        r = x.get(lab)
        if not r or r["clips"] < MIN_CLIPS:
            return {}
        return {k: v["score"] for k, v in r["areas"].items() if v["high"] < 0 and v["score"] < -FLAW}

    cands = []
    for x in out["seeds"]:
        rf = x.get("fresh")
        if not rf:
            continue
        wf, wo = worse_areas(x, "fresh"), worse_areas(x, "old")
        flaws = [f"{k} {pct(wo[k])} on the old and {pct(wf[k])} on the fresh clips" for k in wf if k in wo]
        remarks = [f"{k} {pct(v)} on the {lab} clips only" for lab, w, other in (("fresh", wf, wo), ("old", wo, wf))
                   for k, v in w.items() if k not in other]
        enough = rf["clips"] >= MIN_CLIPS
        cands.append({"seed": x["seed"], "fresh": rf["score"], "old": x["old"]["score"] if x.get("old") else None,
                      "sure": enough and rf.get("low_strict", rf["low"]) > 0, "plain": enough and rf["low"] > 0,
                      "flaws": flaws, "remarks": remarks})
    if cands:
        cands.sort(key=lambda c: -c["fresh"])
        good = [c for c in cands if c["sure"] and not c["flaws"]]
        flawed = [c for c in cands if c["sure"] and c["flaws"]]
        maybe = [c for c in cands if c["plain"] and not c["sure"]]
        out["seed_pick"] = [c["seed"] for c in good]
        out["seed_pick_flawed"] = [c["seed"] for c in flawed]
        out["seed_pick_maybe"] = [c["seed"] for c in maybe]
        out["seed_remarks"] = {str(c["seed"]): c["remarks"] for c in cands if c["remarks"]}
        out["seed_show"] = [c["seed"] for c in good + flawed + maybe + [c for c in cands if not c["plain"]]][:3]

        def both_txt(c, remarks=True):
            return (f"{c['seed']} (fresh {pct(c['fresh'])}" + (f", old {pct(c['old'])}" if c["old"] is not None else "")
                    + ("; but " + ", ".join(c["remarks"]) if remarks and c["remarks"] else "") + ")")

        print(f"YARD_SEED_PICK better than seed {live} on the fresh clips at the level that allows for {len(cands)} seeds being "
              "tried, and no area clearly worse on both the old and the fresh clips: " + (", ".join(both_txt(c) for c in good) or "none")
              + ("; better but clearly worse in an area: " + ", ".join(f"{c['seed']} ({'; '.join(c['flaws'])})" for c in flawed) if flawed else "")
              + ("; better at the ordinary 95% level only, which one of " + str(len(cands)) + " seeds can reach by luck: "
                 + ", ".join(both_txt(c, False) for c in maybe) if maybe else ""))
        for c in good[:3]:
            g = {}
            for t in GARMENT_TYPES:
                rt = Pairs(table, "fav2", "fav2", [(c["seed"], live)], a.fresh, weights, only=types, kind=t).result(a.boot)
                if rt:
                    g[t] = rt
            if g:
                c["garments"] = g
                print(f"YARD_SEED_GARMENT seed {c['seed']}, fresh clips: "
                      + " | ".join(f"{t} {rng_txt(v)} {word(v)}, {v['clips']} clips" for t, v in g.items()))
    # ---- the control: favourite #2 rendered again at the live seed must be favourite #2
    if (CONTROL, live) in table:
        ctl = {}
        for label, group in (("fresh", a.fresh), ("old", a.old)):
            r = Pairs(table, CONTROL, "fav2", [(live, live)], group, weights).result(a.boot) if group else None
            if r:
                biggest = max((abs(v["score"]) for v in r["areas"].values()), default=0.0)
                ctl[label] = {**r, "largest_area": biggest, "ok": abs(r["score"]) < 0.005 and biggest < 0.02}
        for label, r in ctl.items():
            if r["ok"]:
                verdict = ("the same: a render can be repeated" if label == "fresh" else
                           "the same: the stored videos are what today's renderer makes")
            elif label == "fresh":
                verdict = "NOT the same: two renders of the same thing differ by this much, smaller differences mean nothing"
            elif ctl.get("fresh", {}).get("ok", True):
                verdict = (f"NOT the same: the stored seed {live} videos are not what today's renderer makes; on the old "
                           "clips read the seeds against today's render (YARD_CONTROL_SEEDS)")
            else:
                verdict = "NOT the same, as on the fresh clips: the renderer itself is not repeatable to that extent"
            print(f"YARD_CONTROL favourite #2 rendered again at seed {live} against its {'videos of this run' if label == 'fresh' else 'stored videos'}, "
                  f"{r['clips']} {label} clips: {rng_txt(r)}, largest area {r['largest_area'] * 100:.1f}%: {verdict}")
        if ctl:
            out["control"] = ctl
        sds = seeds_on(a.old)
        r = Pairs(table, "fav2", CONTROL, [(sd, live) for sd in sds], a.old, weights).result(a.boot) if len(sds) >= 2 else None
        if r:
            out.setdefault("control", {})["seeds_old"] = {**r, "seeds": sds}
            print(f"YARD_CONTROL_SEEDS the {len(sds)} other seeds together against favourite #2 rendered again today at seed "
                  f"{live}, old clips: {rng_txt(r)}: {word(r)}")
    # ---- every other model against favourite #2 at the same seeds, all clips
    for model in sorted({m for (m, _) in table if m not in ("fav2", CONTROL)}):
        seeds = sorted(s for (m, s) in table if m == model and ("fav2", s) in table)
        if not seeds:
            continue
        pairs = [(s, s) for s in seeds]
        r = Pairs(table, model, "fav2", pairs, sets, weights).result(a.boot)
        if r is None:
            continue
        half.append((r["high"] - r["low"]) / 2)
        # the fresh clips alone: every video of them is from this run (the old clips' favourite #2 videos are stored ones)
        rfr = Pairs(table, model, "fav2", pairs, a.fresh, weights).result(a.boot) if a.fresh else None
        print(f"YARD_MODEL {model} against favourite #2 at the same seeds ({', '.join(map(str, seeds))}), {r['clips']} clips: "
              f"{rng_txt(r)}: {word(r)}; each seed alone " + ", ".join(pct(x) for x in r["per_seed"])
              + (f"; on the {rfr['clips']} fresh clips alone {rng_txt(rfr)}: {word(rfr)}" if rfr else ""))
        g = {}
        for t in GARMENT_TYPES:
            rt = Pairs(table, model, "fav2", pairs, sets, weights, only=types, kind=t).result(a.boot)
            if rt:
                g[t] = rt
        if g:
            print(f"YARD_MODEL_GARMENT {model}: " + " | ".join(f"{t} {rng_txt(v)} {word(v)}, {v['clips']} clips" for t, v in g.items()))
        print(f"YARD_MODEL_AREAS {model}: " + ", ".join(f"{k} {rng_txt(v)}" for k, v in r["areas"].items()))
        out["models"].append({"model": model, "seeds": seeds, "all": r, "fresh": rfr, "garments": g})
    # the models for the side-by-side video: up to four, the best overall first, by the name of their render at the
    # live seed. Four, not fewer: the score does not see a garment's shape, the owner's eye does.
    names = {(str(meta["model"]), int(meta["seed"])): meta["name"] for _, meta in entries(yard)}
    show = [names[(m["model"], live)] for m in sorted(out["models"], key=lambda m: -m["all"]["score"]) if (m["model"], live) in names]
    if show:
        out["model_show"] = show[:4]
    # ---- the same garments as really worn: the level a perfect try-on would reach
    real: dict = {}
    if a.worn and Path(a.worn).is_dir():
        for st in sets:
            for c, row in raw_values(Path(a.worn) / st, WORN).items():
                real[(st, c)] = row
    real_fab = geo(v["fabric_ratio"] for v in real.values() if v.get("fabric_ratio", 0) > 0)
    if real:
        fav_raw = load_raw(yard, a.baseline, sets)
        txt, res, n = real_against(real, fav_raw)
        if txt:
            out["real"] = {"clips": n, "all": res, "garments": {}}
            print(f"YARD_REAL the same garments as really worn ({n} real videos) next to favourite #2 at seed {live} on the "
                  f"same garments: {txt}")
            for t in GARMENT_TYPES:
                only = {(st, c) for st in sets for c, k in types.get(st, {}).items() if k == t}
                txt, res, n = real_against(real, fav_raw, only)
                if txt:
                    out["real"]["garments"][t] = {"clips": n, **res}
                    print(f"YARD_REAL_GARMENT {t}, {n} real videos: {txt}")
        # does a model come nearer to the real videos where favourite #2 stands apart from them?
        for model, mine in sorted(load_model_raw(yard, sets, live).items()):
            txt, res = gaps_against(real, fav_raw, mine, types)
            if txt:
                out.setdefault("gaps", {})[model] = res
                print(f"YARD_GAPS {model}: {txt}")
    # ---- which way the fabric differs: flatter or more contrasty than the garment photo
    fab = load_fabric(yard, a.baseline, sets, live)
    fav = {sd: geo(v.values()) for (m, sd), v in sorted(fab.items()) if m == "fav2"}
    fav = {sd: x for sd, x in fav.items() if x}
    if fav:
        print("YARD_FABRIC favourite #2: the garment's light-dark spread is "
              + ", ".join(f"x{x:.2f} at seed {sd}" for sd, x in fav.items()) + " of the photo's (1.00 = the same, under 1 = flatter)"
              + (f"; the same garments as really worn: x{real_fab:.2f}" if real_fab else ""))
        out["fabric"] = {"fav2": fav, "models": {}, "real": real_fab}
    for model in sorted({m for (m, _) in fab if m not in ("fav2", CONTROL)}):
        mine, theirs = [], []
        for (m, sd), rows in fab.items():
            if m != model:
                continue
            for key, x in rows.items():
                y = fab.get(("fav2", sd), {}).get(key)
                if y:
                    mine.append(x)
                    theirs.append(y)
        if mine:
            gm, gt = geo(mine), geo(theirs)
            near = ""
            if real_fab:
                dm, dt = abs(float(np.log(gm / real_fab))), abs(float(np.log(gt / real_fab)))
                near = ("; nearer to the real videos than favourite #2" if dm < dt - 0.015 else
                        "; further from the real videos than favourite #2" if dm > dt + 0.015 else
                        "; as near to the real videos as favourite #2")
            out.setdefault("fabric", {"fav2": fav, "models": {}, "real": real_fab})["models"][model] = {"model": gm, "fav2": gt, "pairs": len(mine)}
            print(f"YARD_FABRIC {model}: x{gm:.2f} of the photo's, favourite #2 at the same seeds and clips x{gt:.2f} "
                  f"({len(mine)} renders): " + ("flatter than favourite #2" if gm < gt * 0.985 else
                                               "more contrast than favourite #2" if gm > gt * 1.015 else "the same as favourite #2") + near)
    if UNREADABLE:
        rel = sorted(str(Path(x).relative_to(yard)) if Path(x).is_relative_to(yard) else x for x in UNREADABLE)
        out["unreadable"] = rel
        print(f"YARD_NOTE {len(rel)} score file(s) are there but cannot be read and are left out (start the run again: "
              "they are made again): " + ", ".join(rel[:8]) + (" ..." if len(rel) > 8 else ""))
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
