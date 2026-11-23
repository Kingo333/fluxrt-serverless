"""fxreal9 on real tensors: the keep-spread term (favourite #2's fabric contrast is kept, one-sided) and the floors
that make the garment terms count at the noise levels the live model runs at. Needs torch (skipped without it, e.g.
in CI). The code under test is taken from the patch itself."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "training/patches/flux2-klein-fx9.patch"


def added() -> list[str]:
    return [l[1:] for l in PATCH.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]


def fx9():
    """The helper functions exactly as the patch adds them."""
    src = "\n".join(added())
    code = src[src.index("def _azyah_spread(x, m):"):src.index("    elif args.ref_weight > 0 or args.keep_spread_weight > 0:")]
    ns = {"torch": torch}
    exec(compile(code, "fx9", "exec"), ns)
    return types.SimpleNamespace(spread=ns["_azyah_spread"], keep=ns["_azyah_keep_spread_term"])


def test_spread_is_the_weighted_standard_deviation_per_picture_and_channel():
    f = fx9()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 8, 6, generator=g)
    ones = torch.ones(2, 1, 8, 6)
    assert torch.allclose(f.spread(x, ones), x.flatten(2).std(dim=2, unbiased=False), atol=1e-6)
    # only the weighted area counts: the top half here
    m = torch.zeros(2, 1, 8, 6)
    m[:, :, :4] = 1.0
    assert torch.allclose(f.spread(x, m), x[:, :, :4].flatten(2).std(dim=2, unbiased=False), atol=1e-6)
    # half weights count half (the skin and hair areas of the weight map)
    w = torch.full((2, 1, 8, 6), 0.5)
    assert torch.allclose(f.spread(x, w), f.spread(x, ones), atol=1e-6)
    assert f.spread(x, m).shape == (2, 3)


def test_keep_spread_is_zero_unless_the_add_on_is_flatter_than_favourite_2():
    f = fx9()
    g = torch.Generator().manual_seed(1)
    ref = torch.randn(2, 4, 8, 8, generator=g)
    m = torch.ones(2, 1, 8, 8)
    mean = ref.mean((-2, -1), keepdim=True)
    assert float(f.keep(ref.clone(), ref, m)) == 0.0                       # the same picture
    assert float(f.keep(mean + 1.5 * (ref - mean), ref, m)) == 0.0         # more contrast: allowed
    assert float(f.keep(ref + 3.0, ref, m)) == pytest.approx(0.0, abs=1e-6)     # another colour, the same contrast
    flat = mean + 0.5 * (ref - mean)
    want = (f.spread(ref, m) - f.spread(flat, m)).mean()
    assert float(f.keep(flat, ref, m)) == pytest.approx(float(want), rel=1e-5) and float(want) > 0.2
    # one channel flatter, the others more contrasty: only the flatter one counts
    mixed = mean + torch.tensor([0.5, 2.0, 2.0, 2.0]).view(1, 4, 1, 1) * (ref - mean)
    assert float(f.keep(mixed, ref, m)) == pytest.approx(float((f.spread(ref, m) - f.spread(mixed, m))[:, 0].sum() / 8), rel=1e-5)


def test_keep_spread_pulls_a_flat_prediction_back_and_never_touches_the_reference():
    f = fx9()
    g = torch.Generator().manual_seed(2)
    ref = torch.randn(1, 4, 8, 8, generator=g, requires_grad=True)
    m = torch.ones(1, 1, 8, 8)
    with torch.no_grad():
        start = ref.mean((-2, -1), keepdim=True) + 0.4 * (ref - ref.mean((-2, -1), keepdim=True))
    x = start.clone().requires_grad_(True)
    before = f.spread(x, m).detach().clone()
    loss = f.keep(x, ref.detach(), m)
    loss.backward()
    with torch.no_grad():
        x2 = x - 5.0 * x.grad
    assert torch.all(f.spread(x2, m) > before)                             # every channel gets its contrast back
    assert float(f.keep(x2, ref.detach(), m)) < float(loss)
    # outside the weighted area nothing is asked for
    m2 = torch.zeros(1, 1, 8, 8)
    m2[:, :, :, :4] = 1.0
    y = start.clone().requires_grad_(True)
    f.keep(y, ref.detach(), m2).backward()
    assert float(y.grad[..., 4:].abs().max()) == 0.0 and float(y.grad[..., :4].abs().max()) > 0


def test_the_floors_make_the_terms_count_at_the_live_noise_levels():
    lines = [l.strip() for l in added()]
    snippet = "\n".join(l for l in lines if l.startswith(("_conf_d =", "_conf_p =")))
    assert snippet.count("\n") == 1
    conf = torch.tensor([0.0, 0.12, 0.7])                                  # 1 - noise at noise 1.0, 0.88 and 0.3
    ns = {"args": types.SimpleNamespace(detail_floor=0.5, pair_floor=1.0), "_conf": conf}
    exec(snippet, ns)
    assert ns["_conf_d"].tolist() == pytest.approx([0.5, 0.5, 0.7]) and ns["_conf_p"].tolist() == [1.0, 1.0, 1.0]
    ns = {"args": types.SimpleNamespace(detail_floor=0.0, pair_floor=0.0), "_conf": conf}
    exec(snippet, ns)
    assert ns["_conf_d"] is conf and ns["_conf_p"] is conf                 # off: the very same weights as before
    # the fabric-contrast weights get their floor the same way
    i = lines.index("if args.texture_floor > 0:  # Azyah fxreal9: also at the live model's noise levels")
    assert lines[i + 1] == "_c5 = _c5.clamp(min=args.texture_floor)"
    # and the terms use them
    src = "\n".join(lines)
    assert "_detail = (_conf_d * _per).sum() / _conf_d.sum().clamp(min=1e-6)" in src
    assert "_terms.append(_conf_p[_i] * (_d_hat - _d_true).abs().mean())" in src
