"""Session token format shared by the Cloudflare Worker and the GPU workers.

Token = "<payload>.<hex HMAC-SHA256(SESSION_SIGNING_SECRET, payload)>"

Payload (v2, issued by the Worker from the Premium release on):
    userId.garmentId.tier.maxSessionSeconds.expEpochSeconds

Payload (legacy, issued by the Worker before the Premium release):
    userId.garmentId.expEpochSeconds

`exp` is the last payload segment in both formats, so older servers that read
the last segment as the expiry keep working with v2 tokens.

`exp` is the *connect* deadline. `maxSessionSeconds` caps streaming time once
the model is ready; servers also apply their own MAX_SESSION_SECONDS cap.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional

VALID_TIERS = ("standard", "premium")


@dataclass(frozen=True)
class SessionToken:
    user_id: str
    garment_id: str
    tier: str  # "standard" for legacy tokens
    max_session_s: Optional[int]  # None for legacy tokens
    exp: int
    signature: str
    legacy: bool


def _is_epoch(s: str) -> bool:
    return s.isdigit() and len(s) >= 9


def parse_and_verify(
    token: str,
    secret: str,
    *,
    now: Optional[float] = None,
    allow_legacy: bool = True,
) -> Optional[SessionToken]:
    """Return the decoded token if the signature and expiry are valid, else None.

    Fails closed: an empty secret never validates anything.
    """
    if not secret or not token or not isinstance(token, str) or "." not in token:
        return None

    try:
        payload, sig = token.rsplit(".", 1)
    except ValueError:
        return None

    expected = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig.lower()):
        return None

    parts = payload.split(".")
    now_s = int(now if now is not None else time.time())

    if len(parts) == 5:
        user_id, garment_id, tier, max_s, exp_s = parts
        if tier not in VALID_TIERS or not max_s.isdigit() or not _is_epoch(exp_s):
            return None
        max_session = int(max_s)
        if max_session <= 0:
            return None
        legacy = False
    elif len(parts) == 3 and allow_legacy:
        user_id, garment_id, exp_s = parts
        if not _is_epoch(exp_s):
            return None
        tier, max_session, legacy = "standard", None, True
    else:
        return None

    exp = int(exp_s)
    if exp < now_s:
        return None

    return SessionToken(
        user_id=user_id,
        garment_id=garment_id,
        tier=tier,
        max_session_s=max_session,
        exp=exp,
        signature=sig.lower(),
        legacy=legacy,
    )


def sign(payload: str, secret: str) -> str:
    """Build a token for a payload. Used by tests and admin tooling."""
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


class SessionClock:
    """Tracks the streaming deadline per token, in memory, on one worker.

    The first time a token reaches "ready" its clock starts. Reconnecting with
    the same token resumes the same clock instead of getting a fresh cap.
    A hard ceiling of exp + cap bounds total use of any single token.
    """

    def __init__(self) -> None:
        self._started: dict[str, tuple[float, float]] = {}  # sig -> (started_at, hard_deadline)

    def deadline(self, tok: SessionToken, server_cap_s: int, *, now: Optional[float] = None) -> float:
        now_f = now if now is not None else time.time()
        cap = server_cap_s if tok.max_session_s is None else min(tok.max_session_s, server_cap_s)
        self._prune(now_f)
        started, hard = self._started.setdefault(tok.signature, (now_f, tok.exp + cap))
        return min(started + cap, hard)

    def _prune(self, now_f: float) -> None:
        for sig in [s for s, (_, hard) in self._started.items() if hard < now_f]:
            del self._started[sig]
