"""OpenAI / Azure GPT-Live streaming adapter.

Uses the Live WebSocket protocol (``/v1/live/sessions``), **not** Realtime
(``/v1/realtime``).  Key differences from ``OpenAIRealtimeClient``:

* Auth: Azure ``api-key`` header (or Bearer for OpenAI).
* Lifecycle: ``session.start`` → ``session.started`` → stream audio →
  ``session.close`` → ``session.closed``.
* Audio: ``session.input_audio.append`` (continuous; no commit /
  ``response.create``).
* Captions: ``session.output_transcript.delta`` (assistant text for eval).

Env (standard Azure OpenAI names)::

    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT   # e.g. https://<resource>.openai.azure.com/openai/v1
    GPT_DEPLOYMENT_NAME     # optional model/deployment override

Docs: https://developers.openai.com/api/docs/guides/voice-websockets
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

from realqa.base import (
    ContentDoneEvent,
    ContentTokenEvent,
    ErrorEvent,
    SessionConfig,
    SessionEndEvent,
    SessionReadyEvent,
    StreamEvent,
    StreamingClient,
)

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


DEFAULT_SAMPLE_RATE = 24000
DEFAULT_VOICE = "marin"
DEFAULT_IDLE_AFTER_TRANSCRIPT_S = 2.0
DEFAULT_IDLE_NO_TRANSCRIPT_S = 8.0


class _AsyncRunner:
    """Runs an asyncio event loop in a background daemon thread."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 60):
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def schedule(self, coro):
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)


def _http_to_live_ws(http_base: str) -> str:
    """Convert HTTP endpoint to ``wss://…/live/sessions``."""
    base = http_base.rstrip("/")
    if base.startswith("https://"):
        ws = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        ws = "ws://" + base[len("http://") :]
    elif base.startswith("wss://") or base.startswith("ws://"):
        ws = base
    else:
        ws = "wss://" + base
    path = urlsplit(ws).path.rstrip("/")
    if path.endswith("/live/sessions"):
        return ws if "://" in ws[:6] else ws
    # Endpoint is typically …/openai/v1  →  …/openai/v1/live/sessions
    parts = urlsplit(ws)
    new_path = (parts.path.rstrip("/") or "") + "/live/sessions"
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


class OpenAILiveClient(StreamingClient):
    """GPT-Live WebSocket client (Azure or OpenAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is required for OpenAILiveClient. "
                "Install with: pip install websockets"
            )
        # Dedicated names win over Azure and generic OPENAI_* so GPT Live
        # and GPT Realtime can use different accounts simultaneously; the
        # effective endpoint still decides the auth flavor.
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_LIVE_API_KEY", "")
            or os.environ.get("AZURE_OPENAI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "OPENAI_LIVE_API_KEY (or AZURE_OPENAI_API_KEY / "
                "OPENAI_API_KEY) not set and no api_key provided"
            )
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_LIVE_BASE_URL")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self._default_voice = voice
        self._runner = _AsyncRunner()
        self._event_queue: queue.Queue = queue.Queue()
        self._t0: float = 0.0
        self._audio_started = False
        self._realtime_factor: float = 1.0
        self._connected = False
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._ws: Any = None
        self._recv_task: Any = None
        self._got_text_delta = False
        self._last_transcript_ts: float = 0.0
        self._input_complete = False
        self._session_closed = False
        self._closing = False
        self._lock = threading.Lock()
        self._uses_azure = "azure.com" in (self._base_url or "").lower()

    def connect(self, config: SessionConfig) -> None:
        self._t0 = 0.0
        self._audio_started = False
        self._realtime_factor = float(config.extra.get("realtime_factor", 1.0) or 1.0)
        self._sample_rate = int(config.audio_sample_rate or DEFAULT_SAMPLE_RATE)
        self._got_text_delta = False
        self._last_transcript_ts = 0.0
        self._input_complete = False
        self._session_closed = False
        self._closing = False
        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break
        self._runner.start()
        self._runner.run(self._async_connect(config), timeout=60)
        self._connected = True
        self._event_queue.put(SessionReadyEvent(0.0))

    def _mark_audio_start(self) -> None:
        if not self._audio_started:
            self._t0 = time.monotonic()
            self._audio_started = True

    def _now(self) -> float:
        if not self._audio_started:
            return 0.0
        factor = self._realtime_factor if self._realtime_factor > 0 else 1.0
        return round((time.monotonic() - self._t0) * factor, 3)

    def _auth_headers(self) -> dict[str, str]:
        # Azure deployments use the api-key header; OpenAI-direct uses Bearer
        # (decided by endpoint host, not by which env var holds the key).
        if self._uses_azure:
            return {"api-key": self._api_key}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _async_connect(self, config: SessionConfig) -> None:
        # Effective endpoint decides the flavor: Azure deployments need the
        # api-key header and address the model by deployment name; OpenAI
        # native uses Bearer and the plain model id. AZURE_OPENAI_* /
        # GPT_DEPLOYMENT_NAME apply only when the effective endpoint is
        # Azure, so native works by default (just OPENAI_API_KEY) and a
        # --ws-url override flips the flavor with it.
        base = config.extra.get("ws_url") or self._base_url
        url = _http_to_live_ws(str(base))
        self._uses_azure = "azure.com" in str(base).lower()

        model = (config.model or "gpt-live-1").strip()
        if self._uses_azure:
            dep = os.environ.get("GPT_DEPLOYMENT_NAME", "").strip()
            if dep:
                model = dep

        voice = str(config.extra.get("voice") or self._default_voice)

        connect_kwargs: dict[str, Any] = {
            "additional_headers": self._auth_headers(),
            "max_size": 16_000_000,
            "open_timeout": 30,
        }
        try:
            self._ws = await websockets.connect(url, proxy=None, **connect_kwargs)
        except TypeError:
            self._ws = await websockets.connect(url, **connect_kwargs)

        start_id = f"start_{uuid.uuid4().hex[:10]}"
        session: dict[str, Any] = {
            "model": model,
            "instructions": config.prompt or "",
            "audio": {
                "format": {"type": "audio/pcm", "rate": self._sample_rate},
                "output": {"voice": voice},
            },
        }
        await self._ws.send(json.dumps({
            "type": "session.start",
            "event_id": start_id,
            "session": session,
        }))

        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "session.started":
                break
            if t == "error":
                err = msg.get("error") or msg
                raise RuntimeError(f"session.start error: {err}")

        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for raw_msg in self._ws:
                msg = json.loads(raw_msg)
                event_type = msg.get("type", "")
                ts = self._now()

                if event_type == "session.output_transcript.delta":
                    delta = msg.get("delta", "") or ""
                    if delta:
                        with self._lock:
                            self._got_text_delta = True
                            self._last_transcript_ts = time.time()
                        self._event_queue.put(ContentTokenEvent(ts, delta))
                    continue

                if event_type == "session.input_transcript.delta":
                    # User caption — not used for answer eval.
                    continue

                if event_type == "session.output_audio.delta":
                    # Audio bytes ignored; we score on transcript timing.
                    continue

                if event_type == "session.closed":
                    with self._lock:
                        self._session_closed = True
                    if self._got_text_delta:
                        self._event_queue.put(ContentDoneEvent(ts, ""))
                    self._event_queue.put(SessionEndEvent(ts))
                    return

                if event_type == "error":
                    err = msg.get("error", {})
                    msg_text = (
                        err.get("message", str(err))
                        if isinstance(err, dict)
                        else str(err)
                    )
                    self._event_queue.put(ErrorEvent(ts, msg_text))
                    return

        except asyncio.CancelledError:
            return
        except Exception as exc:
            name = type(exc).__name__
            if "ConnectionClosed" in name or "no close frame" in str(exc):
                if not self._session_closed:
                    self._event_queue.put(SessionEndEvent(self._now()))
                return
            self._event_queue.put(ErrorEvent(self._now(), str(exc)))

    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before send_audio()")
        self._mark_audio_start()
        # PCM16LE: keep even byte length.
        if len(audio_bytes) % 2:
            audio_bytes = audio_bytes[:-1]
        if not audio_bytes:
            return
        event = json.dumps({
            "type": "session.input_audio.append",
            "audio": base64.b64encode(audio_bytes).decode(),
        })
        self._runner.run(self._ws.send(event), timeout=30)

    async def _finalize(self) -> None:
        """Wait for transcript idle (or empty timeout), then ``session.close``."""
        assert self._ws is not None
        idle_ok = DEFAULT_IDLE_AFTER_TRANSCRIPT_S
        idle_empty = DEFAULT_IDLE_NO_TRANSCRIPT_S
        t_wait0 = time.time()
        while True:
            await asyncio.sleep(0.1)
            with self._lock:
                got = self._got_text_delta
                last = self._last_transcript_ts
                closed = self._session_closed
            if closed:
                return
            if got and last > 0 and (time.time() - last) >= idle_ok:
                break
            if not got and (time.time() - t_wait0) >= idle_empty:
                break
            if (time.time() - t_wait0) >= 60.0:
                break

        with self._lock:
            if self._session_closed:
                return
            self._closing = True
        await self._ws.send(json.dumps({
            "type": "session.close",
            "event_id": f"close_{uuid.uuid4().hex[:10]}",
        }))
        for _ in range(150):  # ~15s for session.closed
            await asyncio.sleep(0.1)
            with self._lock:
                if self._session_closed:
                    return
        if not self._session_closed:
            ts = self._now()
            if self._got_text_delta:
                self._event_queue.put(ContentDoneEvent(ts, ""))
            self._event_queue.put(SessionEndEvent(ts))

    def end_audio(self) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before end_audio()")
        self._input_complete = True
        self._runner.run(self._finalize(), timeout=120)

    def iter_events(self, timeout_s: float = 120.0) -> Iterator[StreamEvent]:
        deadline = time.time() + float(timeout_s)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._event_queue.put(
                    ErrorEvent(self._now(), f"iter_events timeout after {timeout_s}s")
                )
                remaining = 0.1
            try:
                event = self._event_queue.get(timeout=min(0.2, max(remaining, 0.05)))
                yield event
                if isinstance(event, (SessionEndEvent, ErrorEvent)):
                    while True:
                        try:
                            yield self._event_queue.get_nowait()
                        except queue.Empty:
                            return
            except queue.Empty:
                if time.time() >= deadline:
                    yield ErrorEvent(
                        self._now(), f"iter_events timeout after {timeout_s}s"
                    )
                    return
                continue

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._runner.run(self._do_close(), timeout=20)
            except Exception:
                pass
        self._runner.stop()
        self._ws = None
        self._recv_task = None
        self._connected = False

    async def _do_close(self) -> None:
        if self._recv_task is not None and not self._recv_task.done():
            # Prefer graceful session.close if not already closing.
            if not self._closing and not self._session_closed and self._ws is not None:
                try:
                    await self._ws.send(json.dumps({
                        "type": "session.close",
                        "event_id": f"close_{uuid.uuid4().hex[:10]}",
                    }))
                    for _ in range(50):
                        await asyncio.sleep(0.1)
                        if self._session_closed:
                            break
                except Exception:
                    pass
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def endpoint(self) -> str:
        """Resolved base URL (non-secret, for run provenance)."""
        return self._base_url
