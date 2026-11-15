"""Skin tone of a person, in six groups, and how to balance a training set across them (fxreal6).

The owner's rule: skin must stay true (no lightening, darkening or smoothing) for everyone. Public
try-on data is mostly light-skinned studio models, so the groups are measured and the rare ones are
shown more often, and every skin number in the evaluation is reported per group.

Tone = ITA (individual typology angle), the standard measure from skin colour science:
ITA = atan2(L* - 50, b*) in degrees, from the median Lab colour of the person's skin pixels.
"""

from __future__ import annotations

import math
from collections import Counter

# (lowest ITA of the group, name), lightest first.
GROUPS = ((55.0, "very_light"), (41.0, "light"), (28.0, "intermediate"), (10.0, "tan"), (-30.0, "brown"), (-math.inf, "dark"))
GROUP_NAMES = tuple(name for _, name in GROUPS)


def ita(lab) -> float:
    """ITA in degrees from a Lab colour (L*, a*, b*)."""
    l, _, b = (float(x) for x in lab)
    return math.degrees(math.atan2(l - 50.0, b))


def group(angle: float) -> str:
    for low, name in GROUPS:
        if angle > low:
            return name
    return GROUPS[-1][1]


def balance_weights(groups: list[str], min_share: float = 0.12, max_repeat: float = 6.0) -> dict[str, float]:
    """How many times more often to show each group so none falls under min_share of the rows shown.
    Groups with nobody stay at 0 (nothing to repeat: that is missing data, reported by missing()).
    No group is repeated more than max_repeat times, so a handful of people are not memorised."""
    counts = Counter(g for g in groups if g in GROUP_NAMES)
    total = sum(counts.values())
    if not total:
        return {}
    weights = {g: 1.0 for g in counts}
    for _ in range(50):  # raise the rare groups until their share holds (others shrink in share)
        shown = sum(counts[g] * weights[g] for g in counts)
        low = [g for g in counts if counts[g] * weights[g] / shown < min_share - 1e-9 and weights[g] < max_repeat]
        if not low:
            break
        for g in low:
            need = min_share * shown / counts[g]
            weights[g] = min(max_repeat, max(weights[g] * 1.05, need))
    return {g: round(w, 3) for g, w in weights.items()}


def shares(groups: list[str], weights: dict[str, float] | None = None) -> dict[str, float]:
    counts = Counter(g for g in groups if g in GROUP_NAMES)
    w = weights or {}
    shown = sum(counts[g] * w.get(g, 1.0) for g in counts)
    return {g: round(counts[g] * w.get(g, 1.0) / shown, 4) if shown else 0.0 for g in GROUP_NAMES}


def missing(groups: list[str], min_people: int = 20) -> list[str]:
    """Groups with too few people to learn from: they need new clips, not more repeats."""
    counts = Counter(groups)
    return [g for g in GROUP_NAMES if counts[g] < min_people]
