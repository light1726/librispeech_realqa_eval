"""Qwen-Audio-3.0-Realtime streaming adapter.

Uses the OpenAI-Realtime-compatible WebSocket API via the ``websockets``
library.  Unlike the ``qwen_omni`` adapter this does **not** depend on
``dashscope`` — just ``websockets``.

Models::

    qwen-audio-3.0-realtime-flash
    qwen-audio-3.0-realtime-plus

Auth: ``DASHSCOPE_API_KEY``.

WebSocket URL (first match wins)::

    SessionConfig.extra["ws_url"]
    QWEN_AUDIO_WS_URL / DASHSCOPE_WS_URL
    DEFAULT_URL (DashScope international)

Pass a private MaaS / China host via ``--ws-url`` or env — do not hardcode it here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import threading
import time
from typing import Any, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


# ── session defaults ──────────────────────────────────────────────

DEFAULT_VOICE = "longanqian"
# Public DashScope international realtime endpoint. Override with
# --ws-url / QWEN_AUDIO_WS_URL / DASHSCOPE_WS_URL / SessionConfig.extra["ws_url"].
# Model is appended as ?model=<id> at connect time.
DEFAULT_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
INTL_URL = DEFAULT_URL  # alias
CN_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_VAD = {
    "type": "server_vad",
    "threshold": 0.5,
    "silence_duration_ms": 500,
}


# ── async helper ──────────────────────────────────────────────────


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


# ── client ────────────────────────────────────────────────────────


class QwenAudioRealtimeClient(StreamingClient):
    """Streaming client for Qwen-Audio-3.0-Realtime via WebSocket.

    Parameters
    ----------
    api_key : str or None
        API key.  Reads ``DASHSCOPE_API_KEY``.
    voice : str
        TTS voice ID (default ``"longanqian"``).
    """

    def __init__(self, api_key: Optional[str] = None, voice: str = DEFAULT_VOICE):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is required for QwenAudioRealtimeClient. "
                "Install with: pip install websockets"
            )
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set and no "
                "api_key provided"
            )
        self._voice = voice
        self._runner = _AsyncRunner()
        self._event_queue: queue.Queue = queue.Queue()
        self._t0: float = 0.0
        self._audio_started = False
        self._realtime_factor: float = 1.0
        self._connected = False
        self._auto_detect = False
        self._input_complete = False
        self._response_active = False
        self._recv_ended = False
        self._settle_s = 2.0
        # Per-session state
        self._ws: Any = None

    # ── lifecycle ──────────────────────────────────────────────

    def connect(self, config: SessionConfig) -> None:
        self._t0 = 0.0
        self._audio_started = False
        self._realtime_factor = float(config.extra.get("realtime_factor", 1.0) or 1.0)
        self._input_complete = False
        self._response_active = False
        self._recv_ended = False
        self._settle_s = float(config.extra.get("settle_s", 2.0) or 2.0)
        self._runner.start()
        self._runner.run(self._async_connect(config))
        self._connected = True
        self._event_queue.put(SessionReadyEvent(0.0))

    def _mark_audio_start(self) -> None:
        """Anchor the timeline at the first outbound audio chunk."""
        if not self._audio_started:
            self._t0 = time.monotonic()
            self._audio_started = True

    def _now(self) -> float:
        if not self._audio_started:
            return 0.0
        factor = self._realtime_factor if self._realtime_factor > 0 else 1.0
        return round((time.monotonic() - self._t0) * factor, 3)

    @staticmethod
    def _ws_url(base_url: str, model: str) -> str:
        """Attach / replace ``model=`` query param without duplicating ``?``."""
        parts = urlsplit(base_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["model"] = model
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )

    async def _async_connect(self, config: SessionConfig) -> None:
        model = config.model or "qwen-audio-3.0-realtime-flash"
        base_url = (
            config.extra.get("ws_url")
            or os.environ.get("QWEN_AUDIO_WS_URL")
            or os.environ.get("DASHSCOPE_WS_URL")
            or DEFAULT_URL
        )
        self._ws_base_url = base_url
        url = self._ws_url(base_url, model)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "x-dashscope-dataInspection": "disable",
        }
        self._ws = await websockets.connect(url, additional_headers=headers)

        voice = config.extra.get("voice", self._voice)
        turn_detection = config.extra.get("turn_detection")
        self._auto_detect = turn_detection is not None
        modalities: list = config.extra.get("modalities", ["text"])
        if turn_detection is not None:
            modalities = ["text", "audio"]  # VAD modes require audio modality
            # Merge defaults (threshold / silence) so CLI `{type: server_vad}`
            # matches the official realtime example.
            if isinstance(turn_detection, dict):
                merged = dict(DEFAULT_VAD)
                merged.update(turn_detection)
                turn_detection = merged
            else:
                turn_detection = {**DEFAULT_VAD, "type": str(turn_detection)}

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": modalities,
                "voice": voice,
                "instructions": config.prompt,
                "turn_detection": turn_detection,
            },
        }
        await self._ws.send(json.dumps(session_config))

        # Start listening for server events in the background
        asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        """Receive events from the WebSocket and enqueue as StreamEvents.

        In VAD / auto-detect mode the server creates one response per
        detected user turn; the loop stays alive between turns while input
        is still streaming so mid-speech answers are all captured.
        """
        try:
            async for raw_msg in self._ws:
                msg = json.loads(raw_msg)
                event_type = msg.get("type", "")
                ts = self._now()

                if not event_type and ("code" in msg or "message" in msg):
                    # DashScope sends some errors (quota, auth) as bare
                    # objects without a type field, then hard-closes the
                    # socket — surface the real reason, not "no close frame".
                    self._event_queue.put(ErrorEvent(
                        ts,
                        f"{msg.get('code', 'error')}: {msg.get('message', '')}",
                    ))
                    return

                if event_type == "response.created":
                    self._response_active = True

                if event_type == "response.audio_transcript.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        self._event_queue.put(ContentTokenEvent(ts, delta))

                elif event_type == "response.audio_transcript.done":
                    transcript = msg.get("transcript", "")
                    self._event_queue.put(ContentDoneEvent(ts, transcript))

                elif event_type == "response.text.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        self._event_queue.put(ContentTokenEvent(ts, delta))

                elif event_type == "response.text.done":
                    text = msg.get("text", "")
                    self._event_queue.put(ContentDoneEvent(ts, text))

                elif event_type == "response.done":
                    self._response_active = False
                    if self._auto_detect and not self._input_complete:
                        continue  # more audio may follow; later turns coming
                    self._event_queue.put(SessionEndEvent(ts))
                    return

                elif event_type == "error":
                    err = msg.get("error", {})
                    msg_text = err.get("message", str(err))
                    self._event_queue.put(ErrorEvent(ts, msg_text))
                    return

        except Exception as exc:
            ts = self._now()
            self._event_queue.put(ErrorEvent(ts, str(exc)))
        finally:
            self._recv_ended = True

    # ── audio streaming ────────────────────────────────────────

    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before send_audio()")
        self._mark_audio_start()
        b64 = base64.b64encode(audio_bytes).decode()
        event = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": b64,
        })
        self._runner.schedule(self._ws.send(event))

    def end_audio(self) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before end_audio()")
        self._input_complete = True
        if not self._auto_detect:
            # Push-to-talk: single response by construction.
            self._runner.run(self._ws.send(json.dumps({"type": "input_audio_buffer.commit"})))
            self._runner.run(self._ws.send(json.dumps({"type": "response.create"})))
            return
        self._runner.run(self._finalize(), timeout=60)

    async def _finalize(self) -> None:
        """Drain an in-flight turn, settle for one late turn, then end.

        The recv loop ends the session itself when a ``response.done``
        arrives after input completion; the forced ``SessionEndEvent``
        below only fires when no further turn is coming.
        """
        for _ in range(200):  # ~20s: active turn drain (fits the 60s end_audio bound)
            await asyncio.sleep(0.1)
            if not self._response_active:
                break
        for _ in range(max(1, int(self._settle_s * 10))):  # settle window
            await asyncio.sleep(0.1)
            if self._response_active:
                break
        for _ in range(200):
            await asyncio.sleep(0.1)
            if not self._response_active:
                break
        if not self._recv_ended:
            self._event_queue.put(SessionEndEvent(self._now()))

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
                self._runner.run(self._do_close(), timeout=10)
            except Exception:
                pass
        self._runner.stop()
        self._ws = None
        self._connected = False

    async def _do_close(self) -> None:
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
        """Resolved WebSocket base URL (non-secret, for run provenance)."""
        return getattr(self, "_ws_base_url", DEFAULT_URL)
