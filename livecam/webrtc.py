"""WebRTC media path for the live try-on (LIVECAM_WEBRTC=1, off by default).

The app keeps its WebSocket for control and signalling. With WebRTC on:
- camera frames arrive on a WebRTC video track;
- the generated frames leave on the same connection.

On RunPod serverless the worker has no inbound ports (the load balancer carries HTTP and
WebSocket only), so the worker reaches the app through a TURN relay.
The relay is used over UDP when this worker can send UDP out, otherwise over TLS on port 443,
which every network lets through like HTTPS.

The TURN provider is not wired in: the app passes the iceServers it got from our backend (a self-hosted
coturn with time-limited REST credentials, or Cloudflare Realtime TURN), and this module only picks
which of those URLs aiortc should use.

FluxRT is unchanged: frames enter the same latest-frame-wins slot as WebSocket frames, and leave
from the same finished-output step (composite, face paste, sharpen), so the model sees the same
288x512 input and the shopper the same 576x1024 output, only carried as video instead of JPEG.

aiortc limits worth knowing (1.16):
- it uses ONE STUN and ONE TURN URL, the first of each, so choose_ice_servers() puts the right ones first;
- it gathers all candidates before answering (no trickle).
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import os
import re
import socket
import struct
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("livecam.webrtc")

try:  # aiortc (and PyAV) are only in images built with livecam/requirements-webrtc.txt
    import av
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from aiortc.rtcrtpsender import RTCRtpSender

    _IMPORT_ERROR: Optional[str] = None
except Exception as e:  # noqa: BLE001 - any import problem means "not available", never a crash
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    MediaStreamTrack = object  # type: ignore[misc,assignment]


def available() -> bool:
    return _IMPORT_ERROR is None


def unavailable_reason() -> Optional[str]:
    return _IMPORT_ERROR


# ---------------------------------------------------------------------------------------------
# Outbound UDP check: a STUN binding request to the relay's own STUN port (coturn and Cloudflare
# both answer STUN without credentials), so the check tests the exact path the media would take.
# ---------------------------------------------------------------------------------------------

STUN_MAGIC_COOKIE = 0x2112A442
# Only used when the app's iceServers name no STUN or UDP TURN host to probe.
STUN_HOST = os.getenv("LIVECAM_WEBRTC_STUN_HOST", "stun.cloudflare.com")
STUN_PORT = 3478


def stun_binding_request(txid: bytes) -> bytes:
    """A 20-byte STUN Binding Request (RFC 8489) with no attributes."""
    if len(txid) != 12:
        raise ValueError("STUN transaction id is 12 bytes")
    return struct.pack("!HHI", 0x0001, 0, STUN_MAGIC_COOKIE) + txid


def is_stun_binding_success(data: bytes, txid: bytes) -> bool:
    if len(data) < 20:
        return False
    msg_type, _length, cookie = struct.unpack("!HHI", data[:8])
    return msg_type == 0x0101 and cookie == STUN_MAGIC_COOKIE and data[8:20] == txid


async def probe_outbound_udp(host: str = STUN_HOST, port: int = STUN_PORT, timeout: float = 1.0, tries: int = 2) -> bool:
    """True when a STUN request over UDP gets an answer: this worker can use TURN over UDP."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_DGRAM)
    except OSError:
        return False
    if not infos:
        return False
    addr = infos[0][4]
    txid = os.urandom(12)
    answered: asyncio.Future = loop.create_future()

    class _Probe(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, _addr: Any) -> None:
            if not answered.done() and is_stun_binding_success(data, txid):
                answered.set_result(True)

        def error_received(self, exc: Exception) -> None:  # ICMP unreachable etc.
            pass

    try:
        transport, _ = await loop.create_datagram_endpoint(_Probe, remote_addr=addr)
    except OSError:
        return False
    try:
        for _ in range(max(1, tries)):
            transport.sendto(stun_binding_request(txid))
            try:
                return bool(await asyncio.wait_for(asyncio.shield(answered), timeout))
            except asyncio.TimeoutError:
                continue
        return False
    finally:
        transport.close()


_UDP_CACHE: Dict[Tuple[str, int], Tuple[bool, float]] = {}
UDP_RECHECK_S = 600.0


def probe_target(ice_servers: Any) -> Tuple[str, int]:
    """Where to send the UDP check: the first STUN URL, else the first TURN-over-UDP URL, of the
    app's iceServers; LIVECAM_WEBRTC_STUN_HOST:3478 when they name neither."""
    turn_udp: Optional[Tuple[str, int]] = None
    for server in ice_servers if isinstance(ice_servers, list) else []:
        urls = server.get("urls") if isinstance(server, dict) else None
        for u in urls if isinstance(urls, list) else [urls]:
            p = parse_ice_url(u) if isinstance(u, str) else None
            if not p or p["port"] == 53:
                continue
            if p["scheme"] == "stun":
                return p["host"], p["port"]
            if p["scheme"] == "turn" and p["transport"] == "udp" and turn_udp is None:
                turn_udp = (p["host"], p["port"])
    return turn_udp or (STUN_HOST, STUN_PORT)


async def outbound_udp_ok(target: Optional[Tuple[str, int]] = None, force: bool = False) -> bool:
    """probe_outbound_udp() to that relay, remembered for 10 minutes (one probe per worker and relay)."""
    host, port = target or (STUN_HOST, STUN_PORT)
    now = time.time()
    cached = _UDP_CACHE.get((host, port))
    if force or cached is None or now - cached[1] > UDP_RECHECK_S:
        ok = await probe_outbound_udp(host, port)
        if cached is None or cached[0] != ok:
            log.info("livecam webrtc: outbound UDP to %s:%d %s", host, port,
                     "works (TURN over UDP)" if ok else "blocked (TURN over TLS 443)")
        _UDP_CACHE[(host, port)] = (ok, now)
        return ok
    return cached[0]


# ---------------------------------------------------------------------------------------------
# ICE servers: aiortc takes the first STUN and the first TURN URL only
# ---------------------------------------------------------------------------------------------

_URL_RE = re.compile(r"^(?P<scheme>stun|stuns|turn|turns):(?P<host>[^?:]+)(?::(?P<port>\d+))?(?:\?transport=(?P<transport>\w+))?$")


def parse_ice_url(url: str) -> Optional[Dict[str, Any]]:
    m = _URL_RE.match(url.strip()) if isinstance(url, str) else None
    if not m:
        return None
    scheme = m.group("scheme")
    port = int(m.group("port")) if m.group("port") else (5349 if scheme in ("turns", "stuns") else 3478)
    transport = (m.group("transport") or ("tcp" if scheme == "turns" else "udp")).lower()
    return {"url": url.strip(), "scheme": scheme, "host": m.group("host"), "port": port, "transport": transport}


def _turn_rank(p: Dict[str, Any], udp_ok: bool) -> Tuple[int, int]:
    """Lower is better. UDP first when it works; otherwise TLS on 443 (passes like HTTPS)."""
    if udp_ok:
        order = {("turn", "udp", 3478): 0, ("turn", "udp", 443): 1, ("turns", "tcp", 443): 2,
                 ("turns", "tcp", 5349): 3, ("turn", "tcp", 3478): 4, ("turn", "tcp", 80): 5}
    else:
        order = {("turns", "tcp", 443): 0, ("turns", "tcp", 5349): 1, ("turn", "tcp", 80): 2,
                 ("turn", "tcp", 3478): 3}
    key = (p["scheme"], p["transport"], p["port"])
    if key in order:
        return (order[key], 0)
    if not udp_ok and p["transport"] == "udp":
        return (99, p["port"])  # UDP is blocked: never pick it
    return (50, p["port"])


def choose_ice_servers(ice_servers: Any, udp_ok: bool) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The app's iceServers (coturn REST credentials or Cloudflare's generate-ice-servers answer),
    reduced to what aiortc uses: one STUN URL (only when UDP works) and the best TURN URL with its
    credentials. Port 53 is dropped (Cloudflare can list it; it times out on many networks).
    Returns (servers, chosen TURN url or None)."""
    stun: List[Dict[str, Any]] = []
    turn: List[Tuple[Dict[str, Any], str, str]] = []
    for server in ice_servers if isinstance(ice_servers, list) else []:
        if not isinstance(server, dict):
            continue
        urls = server.get("urls")
        urls = urls if isinstance(urls, list) else [urls]
        user, cred = server.get("username"), server.get("credential")
        for u in urls:
            p = parse_ice_url(u) if isinstance(u, str) else None
            if not p or p["port"] == 53:
                continue
            if p["scheme"] == "stun":
                stun.append(p)
            elif p["scheme"] in ("turn", "turns") and isinstance(user, str) and isinstance(cred, str) and user and cred:
                turn.append((p, user, cred))
    servers: List[Dict[str, Any]] = []
    if udp_ok and stun:
        servers.append({"urls": [stun[0]["url"]]})
    chosen = None
    usable = [t for t in turn if not (not udp_ok and t[0]["transport"] == "udp")]
    if usable:
        best = min(usable, key=lambda t: _turn_rank(t[0], udp_ok))
        chosen = best[0]["url"]
        servers.append({"urls": [chosen], "username": best[1], "credential": best[2]})
    return servers, chosen


def describe_turn(url: Optional[str]) -> str:
    """Short label for logs and the app's readout: 'udp', 'tls443', 'tcp', ... or 'none'."""
    p = parse_ice_url(url) if url else None
    if not p:
        return "none"
    if p["scheme"] == "turns":
        return f"tls{p['port']}"
    return p["transport"] if p["port"] == 3478 else f"{p['transport']}{p['port']}"


# ---------------------------------------------------------------------------------------------
# Encoder bitrate (aiortc's defaults are tuned for webcams: VP8 at most 1.5 Mbit/s, H.264 3 Mbit/s)
# ---------------------------------------------------------------------------------------------

def configure_bitrates(max_kbps: int, start_kbps: int) -> Dict[str, int]:
    """Raise aiortc's encoder ceiling for the 576x1024 output. The browser's bandwidth estimate
    (REMB) still moves the rate between aiortc's minimum and this ceiling."""
    from aiortc.codecs import h264, vpx

    max_bps = max(500_000, int(max_kbps) * 1000)
    start_bps = max(300_000, min(int(start_kbps) * 1000, max_bps))
    h264.MAX_BITRATE = max_bps
    h264.DEFAULT_BITRATE = start_bps
    vpx.MAX_BITRATE = max_bps
    vpx.DEFAULT_BITRATE = start_bps
    return {"max_bps": max_bps, "start_bps": start_bps}


def prefer_codec(transceiver: Any, mime: str = "video/H264") -> None:
    """Put one codec first (H.264: hardware decode on phones and the kiosk), keep the rest."""
    caps = RTCRtpSender.getCapabilities("video").codecs
    first = [c for c in caps if c.mimeType.lower() == mime.lower()]
    rest = [c for c in caps if c.mimeType.lower() != mime.lower()]
    if first:
        transceiver.setCodecPreferences(first + rest)


# ---------------------------------------------------------------------------------------------
# Tracks and the bridge
# ---------------------------------------------------------------------------------------------

class OutputTrack(MediaStreamTrack):  # type: ignore[misc,valid-type]
    """The generated frames, as a video track. Each new frame goes out as soon as it is pushed;
    with nothing new the last frame is resent once a second so the stream stays alive."""

    kind = "video"

    def __init__(self, keepalive_s: float = 1.0) -> None:
        super().__init__()
        self._frame: Optional[np.ndarray] = None
        self._new = asyncio.Event()
        self._keepalive_s = keepalive_s
        self._t0: Optional[float] = None
        self._last_pts = -1
        self.pushed = 0
        self.sent = 0

    def push(self, rgb: np.ndarray) -> None:
        """One finished frame, RGB uint8 HxWx3 (called on the event loop)."""
        self._frame = rgb
        self.pushed += 1
        self._new.set()

    async def recv(self):  # noqa: D401 - aiortc API
        while self._frame is None:
            self._new.clear()
            await self._new.wait()
        try:
            await asyncio.wait_for(self._new.wait(), timeout=self._keepalive_s)
        except asyncio.TimeoutError:
            pass
        self._new.clear()
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(self._frame), format="rgb24")
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        pts = int((now - self._t0) * 90_000)
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        frame.pts = pts
        frame.time_base = fractions.Fraction(1, 90_000)
        self.sent += 1
        return frame


OnFrame = Callable[[np.ndarray, float], Awaitable[None]]
OnState = Callable[[str], Awaitable[None]]


class RtcBridge:
    """One WebRTC connection for one live session: camera track in, generated track out."""

    def __init__(self, on_frame: OnFrame, on_state: Optional[OnState] = None, prefer: str = "video/H264") -> None:
        if not available():
            raise RuntimeError(f"aiortc unavailable: {_IMPORT_ERROR}")
        self._on_frame = on_frame
        self._on_state = on_state
        self._prefer = prefer
        self.pc: Optional[RTCPeerConnection] = None
        self.output = OutputTrack()
        self.state = "new"
        self.turn: str = "none"
        self.frames_in = 0
        self.last_in_size: Optional[Tuple[int, int]] = None
        self._tasks: List[asyncio.Task] = []
        self._closed = False

    @property
    def live(self) -> bool:
        return self.state == "connected" and not self._closed

    async def accept_offer(self, offer_sdp: str, ice_servers: Any, udp_ok: Optional[bool] = None) -> str:
        """Apply the app's offer and return our answer (all candidates included)."""
        if udp_ok is None:
            udp_ok = await outbound_udp_ok(probe_target(ice_servers))
        servers, chosen = choose_ice_servers(ice_servers, udp_ok)
        self.turn = describe_turn(chosen)
        config = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in servers])
        pc = RTCPeerConnection(configuration=config)
        self.pc = pc

        @pc.on("track")
        def _on_track(track: Any) -> None:
            if track.kind == "video":
                self._tasks.append(asyncio.ensure_future(self._consume(track)))

        @pc.on("connectionstatechange")
        async def _on_change() -> None:
            self.state = pc.connectionState
            log.info("livecam webrtc: %s (turn=%s)", self.state, self.turn)
            if self._on_state is not None:
                try:
                    await self._on_state(self.state)
                except Exception as e:  # noqa: BLE001 - a closed socket must not break the bridge
                    log.debug("livecam webrtc: state callback failed: %s", e)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        pc.addTrack(self.output)  # reuses the offered video transceiver (sendrecv)
        for t in pc.getTransceivers():
            if t.kind == "video":
                prefer_codec(t, self._prefer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    async def _consume(self, track: Any) -> None:
        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("livecam webrtc: input frame failed: %s", e)
                continue
            bgr = frame.to_ndarray(format="bgr24")
            self.frames_in += 1
            self.last_in_size = (bgr.shape[1], bgr.shape[0])
            await self._on_frame(bgr, time.time())

    def push_output(self, rgb: np.ndarray) -> None:
        self.output.push(rgb)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.pc is not None:
            try:
                await self.pc.close()
            except Exception:  # noqa: BLE001
                pass
        self.state = "closed"
