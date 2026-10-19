import time

from session_token import SessionClock, parse_and_verify, sign

SECRET = "test-secret"


def future(s=300):
    return int(time.time()) + s


def test_v2_token_round_trip():
    tok = parse_and_verify(sign(f"u1.g1.premium.300.{future()}", SECRET), SECRET)
    assert tok and tok.tier == "premium" and tok.max_session_s == 300 and not tok.legacy


def test_legacy_token_is_standard():
    tok = parse_and_verify(sign(f"u1.g1.{future()}", SECRET), SECRET)
    assert tok and tok.tier == "standard" and tok.legacy and tok.max_session_s is None


def test_legacy_can_be_refused():
    assert parse_and_verify(sign(f"u1.g1.{future()}", SECRET), SECRET, allow_legacy=False) is None


def test_missing_expiry_rejected():
    assert parse_and_verify(sign("u1.g1", SECRET), SECRET) is None
    assert parse_and_verify(sign("u1.g1.notanumber", SECRET), SECRET) is None
    assert parse_and_verify(sign("u1.g1.standard.300.123", SECRET), SECRET) is None  # too short to be epoch


def test_expired_rejected():
    assert parse_and_verify(sign(f"u1.g1.standard.300.{future(-10)}", SECRET), SECRET) is None


def test_forged_rejected():
    good = sign(f"u1.g1.standard.300.{future()}", SECRET)
    assert parse_and_verify(good, "other-secret") is None
    tampered = good.replace("standard", "premium", 1)
    assert parse_and_verify(tampered, SECRET) is None


def test_empty_secret_fails_closed():
    assert parse_and_verify(sign(f"u1.g1.{future()}", ""), "") is None


def test_bad_tier_and_cap_rejected():
    assert parse_and_verify(sign(f"u1.g1.gold.300.{future()}", SECRET), SECRET) is None
    assert parse_and_verify(sign(f"u1.g1.standard.0.{future()}", SECRET), SECRET) is None


def test_clock_caps_and_resumes():
    now = 1_900_000_000
    tok = parse_and_verify(sign(f"u1.g1.standard.300.{now + 300}", SECRET), SECRET, now=now)
    clock = SessionClock()
    d1 = clock.deadline(tok, 300, now=now)
    assert d1 == now + 300
    # Reconnect later with the same token: same clock, no fresh 5 minutes.
    assert clock.deadline(tok, 300, now=now + 200) == d1
    # Server cap lower than token cap wins.
    tok2 = parse_and_verify(sign(f"u2.g1.standard.900.{now + 300}", SECRET), SECRET, now=now)
    assert clock.deadline(tok2, 300, now=now) == now + 300
