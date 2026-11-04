"""Qwen3-Omni-Flash-Realtime streaming API adapter.

Wraps ``dashscope.audio.qwen_omni``, captures per-token transcript
deltas, and emits the common ``StreamEvent`` family.

Timestamps are wall-clock relative to the moment the first audio chunk
is streamed, so input and output share a single real-time timeline.
"""

from __future__ import annotations

import base64
import os
import queue
import time
from typing import Iterator, List, Optional

from realqa.base import (
    AudioOutputEvent,
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
    from dashscope.audio.qwen_omni import (
        AudioFormat,
        MultiModality,
        OmniRealtimeCallback,
        OmniRealtimeConversation,
    )
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False


# ------------------------------------------------------------------
# internal callback
# ------------------------------------------------------------------


if HAS_DASHSCOPE:
    class _QwenCallback(OmniRealtimeCallback):
        """Bridges dashscope events to the common ``StreamEvent`` family.

        Events are enqueued into a thread-safe queue so ``iter_events`` can
        drain them without a busy loop.
        """

        def __init__(self, event_queue: queue.Queue, client: "QwenOmniRealtimeClient"):
            super().__init__()
            self._queue = event_queue
            self._client = client

        def _ts(self) -> float:
            """Wall-clock seconds since the first audio chunk was sent."""
            return self._client._now()

        # pylint: disable=invalid-name
        def on_event(self, response):
            t = response.get("type", "")
            ts = self._ts()
            client = self._client
            if t == "response.audio_transcript.delta":
                token = response.get("transcript", "")
                if token:
                    client._response_active = True
                    self._queue.put(ContentTokenEvent(ts, token))
            elif t == "response.audio_transcript.done":
                transcript = response.get("transcript", "")
                self._queue.put(ContentDoneEvent(ts, transcript))
            elif t == "response.audio.done":
                self._queue.put(AudioOutputEvent(ts))
            elif t == "response.done":
                client._response_active = False
                if client._auto_detect and not client._input_complete:
                    # More audio may follow; keep the callback alive so
                    # later VAD turns are captured.
                    return
                client._session_ended = True
                self._queue.put(SessionEndEvent(ts))
            elif t == "error":
                msg = str(response.get("message", response))
                client._session_ended = True
                self._queue.put(ErrorEvent(ts, msg))
else:
    _QwenCallback = None  # type: ignore[misc, assignment]

# ------------------------------------------------------------------
# client
# ------------------------------------------------------------------


class QwenOmniRealtimeClient(StreamingClient):
    """Streaming client for Qwen3-Omni-Flash-Realtime via dashscope.

    Parameters
    ----------
    api_key : str or None
        DashScope API key.  Reads from ``DASHSCOPE_API_KEY`` env var
        if not provided.
    voice : str
        TTS voice (default ``"Ethan"``).
    """

    def __init__(self, api_key: Optional[str] = None, voice: str = "Ethan"):
        if not HAS_DASHSCOPE:
            raise ImportError(
                "dashscope is required for QwenOmniRealtimeClient. "
                "Install with: pip install dashscope"
            )

        import os

        import dashscope

        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set and no api_key provided"
            )
        dashscope.api_key = self._api_key
        self._voice = voice
        self._conv: Optional[OmniRealtimeConversation] = None
        self._queue: queue.Queue = queue.Queue()
        self._t0: float = 0.0
        self._audio_started = False
        self._realtime_factor: float = 1.0
        self._connected = False
        self._auto_detect = False
        self._input_complete = False
        self._response_active = False
        self._session_ended = False
        self._settle_s = 2.0

    def _now(self) -> float:
        if not self._audio_started:
            return 0.0
        factor = self._realtime_factor if self._realtime_factor > 0 else 1.0
        return round((time.monotonic() - self._t0) * factor, 3)

    def _mark_audio_start(self) -> None:
        if not self._audio_started:
            self._t0 = time.monotonic()
            self._audio_started = True

    # -- lifecycle -------------------------------------------------

    def connect(self, config: SessionConfig) -> None:
        self._t0 = 0.0
        self._audio_started = False
        self._realtime_factor = float(config.extra.get("realtime_factor", 1.0) or 1.0)
        self._input_complete = False
        self._response_active = False
        self._session_ended = False
        self._settle_s = float(config.extra.get("settle_s", 2.0) or 2.0)
        cb = _QwenCallback(self._queue, self)

        # Same URL precedence as the qwen_audio adapter: explicit override,
        # then env, then the public DashScope international endpoint.
        url = (
            config.extra.get("ws_url")
            or os.getenv("QWEN_AUDIO_WS_URL")
            or os.getenv("DASHSCOPE_WS_URL")
            or "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
        )
        self._conv = OmniRealtimeConversation(
            model=config.model or "qwen3.5-omni-flash-realtime",
            callback=cb,
            url=url,
        )
        self._conv.connect()

        modalities = [MultiModality.AUDIO, MultiModality.TEXT]
        extra = config.extra
        turn_detection = extra.get("turn_detection")
        self._auto_detect = turn_detection is not None
        session_kwargs = {
            "output_modalities": modalities,
            "voice": extra.get("voice", self._voice),
            "instructions": config.prompt,
        }
        if turn_detection is not None:
            # DashScope Omni: server_vad (default) or other types via SDK.
            td_type = (
                turn_detection.get("type", "server_vad")
                if isinstance(turn_detection, dict)
                else str(turn_detection)
            )
            silence_ms = int(float(extra.get("silence_padding_s", 1.5)) * 1000)
            # API allows [200, 6000]; clamp padding-driven silence into range.
            silence_ms = max(200, min(6000, silence_ms))
            session_kwargs.update(
                enable_turn_detection=True,
                turn_detection_type=td_type,
                turn_detection_silence_duration_ms=silence_ms,
            )
        else:
            # Explicit push-to-talk: client commit + create_response.
            session_kwargs["enable_turn_detection"] = False

        self._conv.update_session(**session_kwargs)
        self._connected = True
        self._queue.put(SessionReadyEvent(0.0))

    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        if self._conv is None:
            raise RuntimeError("connect() must be called before send_audio()")
        self._mark_audio_start()
        encoded = base64.b64encode(audio_bytes).decode()
        self._conv.append_audio(encoded)

    def end_audio(self) -> None:
        if self._conv is None:
            raise RuntimeError("connect() must be called before end_audio()")
        self._input_complete = True
        if not self._auto_detect:
            # Push-to-talk: single response by construction.
            self._conv.commit()
            self._conv.create_response()
            return
        self._drain_and_end()

    def _drain_and_end(self) -> None:
        """Thread-based finalize: drain an active turn, settle for one late
        turn, then force the session end if the callback never did."""
        def _wait(pred, bound_s: float) -> bool:
            for _ in range(max(1, int(bound_s * 10))):
                time.sleep(0.1)
                if pred():
                    return True
            return False

        _wait(lambda: not self._response_active, 30.0)
        _wait(lambda: self._response_active, self._settle_s)
        _wait(lambda: not self._response_active, 30.0)
        if not self._session_ended:
            self._session_ended = True
            self._queue.put(SessionEndEvent(self._now()))

    def iter_events(self, timeout_s: float = 120.0) -> Iterator[StreamEvent]:
        """Drain the event queue until the session ends (bounded)."""
        deadline = time.time() + float(timeout_s)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._queue.put(
                    ErrorEvent(self._now(), f"iter_events timeout after {timeout_s}s")
                )
                remaining = 0.1
            try:
                event = self._queue.get(timeout=min(0.2, max(remaining, 0.05)))
                yield event
                if isinstance(event, (SessionEndEvent, ErrorEvent)):
                    # Drain any remaining events before returning
                    while True:
                        try:
                            remaining_ev = self._queue.get_nowait()
                            yield remaining_ev
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
        if self._conv is not None:
            try:
                self._conv.close()
            except Exception:
                pass
            self._conv = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
