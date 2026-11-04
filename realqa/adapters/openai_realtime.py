"""OpenAI Realtime API (GA) streaming adapter.

Auth/endpoints use the standard ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``.

Uses the GA session shape (``session.type``, ``output_modalities``,
``session.audio.input``) and GA event names
(``response.output_text.delta``, …).  Does **not** send the beta header.

Fixes vs early draft
--------------------
* Bypass HTTP proxy on the WebSocket (proxy=None).
* VAD finalize: after client silence pad, if the server never auto-creates
  a response, fall back to ``response.create`` (prevents infinite hang).
* ``iter_events(timeout_s=…)`` so a stuck session cannot block forever.
* Ignore ``*.done`` full-text when deltas already arrived (timeline hygiene).
* Track ``speech_started`` / ``speech_stopped`` for VAD diagnostics.

Models (examples)::

    gpt-realtime-mini
    gpt-realtime-1.5
    gpt-realtime-2.1
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


DEFAULT_HTTP_BASE = "https://api.openai.com"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SERVER_VAD = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    # Shorter silence → earlier turn cuts on continuous speech.
    "silence_duration_ms": 300,
    "create_response": True,
    "interrupt_response": True,
}
DEFAULT_SEMANTIC_VAD = {
    "type": "semantic_vad",
    # high ≈ ~2s max wait; most eager streaming turn-taking.
    "eagerness": "high",
    "create_response": True,
    "interrupt_response": True,
}
DEFAULT_VAD = DEFAULT_SERVER_VAD


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


def _http_to_ws_base(http_base: str) -> str:
    """Convert ``https://host[/…]`` to ``wss://host/v1/realtime``."""
    base = http_base.rstrip("/")
    if base.startswith("https://"):
        ws = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        ws = "ws://" + base[len("http://") :]
    elif base.startswith("wss://") or base.startswith("ws://"):
        ws = base
    else:
        ws = "wss://" + base
    if ws.rstrip("/").endswith("/realtime"):
        return ws
    return ws.rstrip("/") + "/v1/realtime"


class OpenAIRealtimeClient(StreamingClient):
    """GA OpenAI Realtime WebSocket client (ClosedAI / OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is required for OpenAIRealtimeClient. "
                "Install with: pip install websockets"
            )
        # Dedicated names win over the generic OPENAI_* pair so GPT
        # Realtime and GPT Live can use different accounts simultaneously.
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_REALTIME_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "OPENAI_REALTIME_API_KEY (or OPENAI_API_KEY) not set and no "
                "api_key provided"
            )
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_REALTIME_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_HTTP_BASE
        )
        self._runner = _AsyncRunner()
        self._event_queue: queue.Queue = queue.Queue()
        self._t0: float = 0.0
        self._audio_started = False
        self._realtime_factor: float = 1.0
        self._connected = False
        self._auto_detect = False
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._ws: Any = None
        self._recv_task: Any = None
        self._got_text_delta = False
        self._response_started = False
        self._response_finished = False
        self._response_active = False
        self._speech_started = False
        self._input_complete = False
        self._settle_s = 2.0
        self._recv_ended = False
        self._lock = threading.Lock()

    def connect(self, config: SessionConfig) -> None:
        self._t0 = 0.0
        self._audio_started = False
        self._realtime_factor = float(config.extra.get("realtime_factor", 1.0) or 1.0)
        self._sample_rate = int(config.audio_sample_rate or DEFAULT_SAMPLE_RATE)
        self._got_text_delta = False
        self._response_started = False
        self._response_finished = False
        self._response_active = False
        self._speech_started = False
        self._input_complete = False
        self._settle_s = float(config.extra.get("settle_s", 2.0) or 2.0)
        self._recv_ended = False
        # Drain any leftover events from a prior sample.
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

    @staticmethod
    def _ws_url(base_url: str, model: str) -> str:
        parts = urlsplit(_http_to_ws_base(base_url))
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["model"] = model
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )

    async def _async_connect(self, config: SessionConfig) -> None:
        model = config.model or "gpt-realtime-mini"
        base = config.extra.get("ws_url") or self._base_url
        url = self._ws_url(base, model)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        connect_kwargs: dict[str, Any] = {
            "additional_headers": headers,
            "max_size": 16_000_000,
            "open_timeout": 30,
        }
        try:
            self._ws = await websockets.connect(url, proxy=None, **connect_kwargs)
        except TypeError:
            self._ws = await websockets.connect(url, **connect_kwargs)

        raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
        created = json.loads(raw)
        if created.get("type") == "error":
            err = created.get("error") or created
            raise RuntimeError(f"session.created error: {err}")

        turn_detection = config.extra.get("turn_detection")
        self._auto_detect = turn_detection is not None
        if self._auto_detect:
            if isinstance(turn_detection, dict):
                td_type = str(turn_detection.get("type") or "server_vad")
                overrides = dict(turn_detection)
            else:
                td_type = str(turn_detection)
                overrides = {"type": td_type}
            # Map Qwen-style smart_turn → OpenAI semantic_vad.
            if td_type in ("smart_turn", "semantic"):
                td_type = "semantic_vad"
                overrides["type"] = "semantic_vad"
            base_td = (
                DEFAULT_SEMANTIC_VAD
                if td_type == "semantic_vad"
                else DEFAULT_SERVER_VAD
            )
            td: Any = {**base_td, **overrides}
        else:
            td = None

        session: dict[str, Any] = {
            "type": "realtime",
            "model": model,
            "instructions": config.prompt or "",
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self._sample_rate,
                    },
                    "turn_detection": td,
                },
            },
        }
        await self._ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=30))
            t = msg.get("type")
            if t == "session.updated":
                break
            if t == "error":
                err = msg.get("error") or msg
                raise RuntimeError(f"session.update error: {err}")

        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for raw_msg in self._ws:
                msg = json.loads(raw_msg)
                event_type = msg.get("type", "")
                ts = self._now()

                if event_type == "input_audio_buffer.speech_started":
                    self._speech_started = True
                    continue
                if event_type == "input_audio_buffer.speech_stopped":
                    continue

                if event_type == "response.created":
                    with self._lock:
                        self._response_started = True
                        self._response_active = True
                    continue

                if event_type in (
                    "response.output_text.delta",
                    "response.text.delta",
                    "response.output_audio_transcript.delta",
                    "response.audio_transcript.delta",
                ):
                    delta = msg.get("delta", "")
                    if delta:
                        with self._lock:
                            self._got_text_delta = True
                            self._response_started = True
                        self._event_queue.put(ContentTokenEvent(ts, delta))

                elif event_type in (
                    "response.output_text.done",
                    "response.text.done",
                ):
                    # Only emit done-as-content when no deltas were streamed
                    # (otherwise TimelineBuilder would duplicate the full text).
                    text = msg.get("text", "") or ""
                    if text and not self._got_text_delta:
                        self._event_queue.put(ContentDoneEvent(ts, text))
                    else:
                        self._event_queue.put(ContentDoneEvent(ts, ""))

                elif event_type in (
                    "response.output_audio_transcript.done",
                    "response.audio_transcript.done",
                ):
                    transcript = msg.get("transcript", "") or ""
                    if transcript and not self._got_text_delta:
                        self._event_queue.put(ContentDoneEvent(ts, transcript))
                    else:
                        self._event_queue.put(ContentDoneEvent(ts, ""))

                elif event_type == "response.done":
                    # Prefer any leftover text from the completed response object
                    # when no deltas arrived.
                    if not self._got_text_delta:
                        resp = msg.get("response") or {}
                        for item in (resp.get("output") or []):
                            for c in (item.get("content") or []):
                                piece = c.get("text") or c.get("transcript") or ""
                                if piece:
                                    self._event_queue.put(ContentDoneEvent(ts, piece))
                    with self._lock:
                        self._response_finished = True
                        self._response_started = True
                        self._response_active = False
                    # In duplex/VAD mode a response may arrive mid-stream. Keep the
                    # recv loop alive until input is complete so later turns can
                    # still be captured (true streaming processing).
                    if self._auto_detect and not self._input_complete:
                        # Reset delta flag so a later turn can stream fresh text.
                        self._got_text_delta = False
                        continue
                    self._event_queue.put(SessionEndEvent(ts))
                    return

                elif event_type == "error":
                    err = msg.get("error", {})
                    msg_text = (
                        err.get("message", str(err))
                        if isinstance(err, dict)
                        else str(err)
                    )
                    # Non-fatal: empty buffer / unknown param after we already
                    # have a response — keep waiting for response.done.
                    code = err.get("code") if isinstance(err, dict) else None
                    if code in (
                        "input_audio_buffer_commit_empty",
                        "conversation_already_has_active_response",
                    ):
                        continue
                    self._event_queue.put(ErrorEvent(ts, msg_text))
                    return

        except asyncio.CancelledError:
            return
        except Exception as exc:
            name = type(exc).__name__
            if "ConnectionClosed" in name or "no close frame" in str(exc):
                if not self._response_finished:
                    self._event_queue.put(SessionEndEvent(self._now()))
                return
            self._event_queue.put(ErrorEvent(self._now(), str(exc)))
        finally:
            self._recv_ended = True

    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before send_audio()")
        self._mark_audio_start()
        event = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_bytes).decode(),
        })
        self._runner.run(self._ws.send(event), timeout=30)

    async def _create_response(self) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps({
            "type": "response.create",
            "response": {"output_modalities": ["text"]},
        }))

    async def _poll_state(self, pred, bound_s: float) -> bool:
        """Poll a (lock-guarded) predicate every 100 ms, bounded."""
        for _ in range(max(1, int(bound_s * 10))):
            await asyncio.sleep(0.1)
            with self._lock:
                if pred():
                    return True
        return False

    async def _vad_finalize(self) -> None:
        """After the silence pad: drain active/late turns, then end.

        The evaluated system (model + its server turn-taking policy) decides
        when to respond — the client never forces a response. We (1) drain a
        response that is streaming right now, (2) wait a short settle window
        for one more late turn and drain it too. A model that never chose to
        speak stays silent and is scored as such.
        """
        await self._poll_state(lambda: not self._response_active, 15.0)
        await self._poll_state(lambda: self._response_active, self._settle_s)
        await self._poll_state(lambda: not self._response_active, 15.0)

    def end_audio(self) -> None:
        if self._ws is None:
            raise RuntimeError("connect() must be called before end_audio()")
        if self._auto_detect:
            # Recv loop ends the session at the next response.done now that
            # input is complete.
            self._input_complete = True
            self._runner.run(self._vad_finalize(), timeout=60)
            # A last response.done may race the finalize return; a duplicate
            # SessionEndEvent is drained harmlessly by iter_events.
            if not self._recv_ended:
                self._event_queue.put(SessionEndEvent(self._now()))
            return
        self._input_complete = True
        self._runner.run(
            self._ws.send(json.dumps({"type": "input_audio_buffer.commit"})),
            timeout=30,
        )
        self._runner.run(self._create_response(), timeout=30)

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
                self._runner.run(self._do_close(), timeout=15)
            except Exception:
                pass
        self._runner.stop()
        self._ws = None
        self._recv_task = None
        self._connected = False

    async def _do_close(self) -> None:
        if self._recv_task is not None and not self._recv_task.done():
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
