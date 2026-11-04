"""Gemini Live API streaming adapter.

Uses ``google-genai`` ``client.aio.live.connect`` (no mic/camera deps).

Default model: ``gemini-3.1-flash-live-preview``.

For native-audio Live models the session uses ``AUDIO`` response modality
plus ``output_audio_transcription`` so we still get timed text for eval.
Set ``extra["response_modality"] = "TEXT"`` when the model supports text-only
responses.

Auth: ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from typing import Any, Iterator, List, Optional

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
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None  # type: ignore
    types = None  # type: ignore


DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Zephyr"


class _AsyncRunner:
    """Runs an asyncio event loop in a background daemon thread."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 120):
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


class GeminiLiveClient(StreamingClient):
    """Streaming client for Gemini Live (audio in → text/transcript out)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
    ):
        if not HAS_GENAI:
            raise ImportError(
                "google-genai is required for GeminiLiveClient. "
                "Install with: pip install google-genai"
            )
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set and no api_key "
                "provided"
            )
        self._voice = voice
        self._runner = _AsyncRunner()
        self._event_queue: queue.Queue = queue.Queue()
        self._t0: float = 0.0
        self._audio_started = False
        self._realtime_factor: float = 1.0
        self._connected = False

        self._client: Any = None
        self._session: Any = None
        self._session_cm: Any = None
        self._recv_task: Any = None
        self._sample_rate: int = 16000
        self._input_complete = False
        self._in_turn = False
        self._settle_s = 2.0

    # ── lifecycle ──────────────────────────────────────────────

    def connect(self, config: SessionConfig) -> None:
        self._t0 = 0.0
        self._audio_started = False
        self._realtime_factor = float(config.extra.get("realtime_factor", 1.0) or 1.0)
        self._sample_rate = int(config.audio_sample_rate or 16000)
        self._input_complete = False
        self._in_turn = False
        self._settle_s = float(config.extra.get("settle_s", 2.0) or 2.0)
        self._runner.start()
        self._runner.run(self._async_connect(config))
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
    def _normalize_model(model: str) -> str:
        m = (model or DEFAULT_MODEL).strip()
        if m.startswith("models/"):
            m = m[len("models/") :]
        return m or DEFAULT_MODEL

    def _build_live_config(self, config: SessionConfig) -> Any:
        modality = str(
            config.extra.get("response_modality")
            or config.extra.get("modality")
            or "AUDIO"
        ).upper()
        voice = config.extra.get("voice", self._voice)

        kwargs: dict = {}
        if config.prompt:
            kwargs["system_instruction"] = config.prompt

        if modality == "TEXT":
            kwargs["response_modalities"] = ["TEXT"]
        else:
            # Native-audio Live models: AUDIO + transcription for eval text.
            kwargs["response_modalities"] = ["AUDIO"]
            kwargs["output_audio_transcription"] = {}
            kwargs["speech_config"] = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            )
            thinking = config.extra.get("thinking_level")
            if thinking:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking
                )

        return types.LiveConnectConfig(**kwargs)

    async def _async_connect(self, config: SessionConfig) -> None:
        model = self._normalize_model(config.model)
        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1beta"},
        )
        live_cfg = self._build_live_config(config)
        self._session_cm = self._client.aio.live.connect(model=model, config=live_cfg)
        self._session = await self._session_cm.__aenter__()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        """Consume model turns and map them to StreamEvents.

        Full-duplex protocol: the model may answer mid-stream any number of
        times. Each ``receive()`` call covers one turn; after
        ``turn_complete`` we keep listening for the next turn while input is
        still streaming, and end the session on the first turn that
        completes after input is done.
        """
        try:
            assert self._session is not None
            while True:
                turn_parts: List[str] = []
                turn_done = False
                async for response in self._session.receive():
                    self._in_turn = True
                    ts = self._now()
                    text = getattr(response, "text", None)
                    if text:
                        turn_parts.append(text)
                        self._event_queue.put(ContentTokenEvent(ts, text))
                        continue

                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        continue

                    out_tr = getattr(sc, "output_transcription", None)
                    if out_tr is not None:
                        piece = getattr(out_tr, "text", None) or ""
                        if piece:
                            turn_parts.append(piece)
                            self._event_queue.put(ContentTokenEvent(ts, piece))

                    if getattr(sc, "turn_complete", False):
                        turn_done = True
                        break

                ts = self._now()
                full = "".join(turn_parts)
                if turn_done:
                    if full:
                        self._event_queue.put(ContentDoneEvent(ts, full))
                    if self._input_complete:
                        self._event_queue.put(SessionEndEvent(ts))
                        return
                    self._in_turn = False
                    continue  # listen for the next turn

                # Iterator ended without turn_complete: stream closed.
                if full:
                    self._event_queue.put(ContentDoneEvent(ts, full))
                self._event_queue.put(SessionEndEvent(ts))
                return
        except Exception as exc:
            self._event_queue.put(ErrorEvent(self._now(), str(exc)))

    # ── audio streaming ────────────────────────────────────────

    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        if self._session is None:
            raise RuntimeError("connect() must be called before send_audio()")
        self._mark_audio_start()
        mime = f"audio/pcm;rate={self._sample_rate}"
        self._runner.schedule(self._send_audio(audio_bytes, mime))

    async def _send_audio(self, audio_bytes: bytes, mime: str) -> None:
        assert self._session is not None
        await self._session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type=mime)
        )

    def end_audio(self) -> None:
        if self._session is None:
            raise RuntimeError("connect() must be called before end_audio()")
        # Signal end-of-user-audio so VAD / turn-taking can fire a response;
        # the recv loop ends the session at the next turn_complete afterwards.
        self._input_complete = True
        self._runner.run(
            self._session.send_realtime_input(audio_stream_end=True),
            timeout=60,
        )
        self._runner.run(self._finalize(), timeout=60)

    async def _finalize(self) -> None:
        """Drain the in-flight turn after ``audio_stream_end``, settle for
        one more late turn, then cancel a blocked ``receive()`` so the run
        cannot hang waiting for a turn that will never come."""
        def _recv_done() -> bool:
            return self._recv_task is None or self._recv_task.done()

        for _ in range(200):  # ~20s: drain an active turn (fits the 60s end_audio bound)
            await asyncio.sleep(0.1)
            if _recv_done():
                return
            if not self._in_turn:
                break
        for _ in range(max(1, int(self._settle_s * 10))):  # settle window
            await asyncio.sleep(0.1)
            if _recv_done():
                return
            if self._in_turn:
                break
        for _ in range(200):
            await asyncio.sleep(0.1)
            if _recv_done():
                return
            if not self._in_turn:
                break
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        self._event_queue.put(SessionEndEvent(self._now()))

    # ── event consumption ──────────────────────────────────────

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
        if self._session_cm is not None:
            try:
                self._runner.run(self._do_close(), timeout=30)
            except Exception:
                pass
        self._runner.stop()
        self._session = None
        self._session_cm = None
        self._client = None
        self._connected = False

    async def _do_close(self) -> None:
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def endpoint(self) -> str:
        """Endpoint is managed by the google-genai SDK."""
        return "googleapis:generativelanguage (google-genai SDK)"
