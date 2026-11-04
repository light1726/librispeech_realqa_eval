"""Abstract base layer for streaming speech-to-text API clients.

Defines the common types (events, config, timeline result) and the
``StreamingClient`` ABC that every API adapter must implement.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


# ── session configuration ──────────────────────────────────────────


@dataclass
class SessionConfig:
    """Model-agnostic session parameters passed to every adapter.

    Attributes
    ----------
    model : str
        Model identifier (e.g. ``"qwen3.5-omni-flash-realtime"``).
    prompt : str
        Natural-language instruction sent to the model.
    chunks_per_second : float
        Audio frame / group rate in Hz.  For API inference this should
        match the streaming chunk size: ``1000 / chunk_ms`` (e.g. 5.0
        for 200 ms chunks).
    audio_sample_rate : int
        Expected audio sample rate in Hz (usually 16 000).
    audio_encoding : str
        Audio encoding understood by the API (e.g. ``"pcm16"``).
    extra : dict
        Model-specific key-value overrides (voice, modalities, temperature,
        etc.).  Interpreted by the adapter.
    """

    model: str = ""
    prompt: str = ""
    chunks_per_second: float = 5.0
    audio_sample_rate: int = 16000
    audio_encoding: str = "pcm16"
    extra: Dict[str, Any] = field(default_factory=dict)


# ── timeline result ────────────────────────────────────────────────


@dataclass
class TimelineResult:
    """Output of ``TimelineBuilder`` – a completed token grid.

    Attributes
    ----------
    tokens : list[str]
        Token sequence at *chunks_per_second* resolution.  Each position is either
        a silence token, a content token, or an EOS token.
    content_runs : list[dict]
        Each dict has ``text``, ``t_start``, ``t_end``, ``start_group``,
        ``end_group`` – describing one contiguous content burst.
    chunks_per_second : float
        Grid resolution in Hz.
    prefix_len : int
        Number of leading positions before audio time zero (default 0).
    """

    tokens: List[str] = field(default_factory=list)
    content_runs: List[Dict[str, Any]] = field(default_factory=list)
    chunks_per_second: float = 5.0
    prefix_len: int = 0


# ── unified event family ───────────────────────────────────────────


class StreamEvent:
    """Base class for all streaming API events.

    Parameters
    ----------
    timestamp : float
        Event time in seconds, audio-aligned.
    """

    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ts={self.timestamp:.3f})"


class SessionReadyEvent(StreamEvent):
    """Session successfully created and ready for audio."""

    pass


class ContentTokenEvent(StreamEvent):
    """One text token emitted by the model during streaming.

    Parameters
    ----------
    token : str
        The token string (may be a sub-word or full word).
    """

    def __init__(self, timestamp: float, token: str):
        super().__init__(timestamp)
        self.token = token

    def __repr__(self) -> str:
        return f"ContentTokenEvent(ts={self.timestamp:.3f}, token={self.token!r})"


class ContentDoneEvent(StreamEvent):
    """Final transcript for the current content burst.

    Falls back as a single emission when per-token deltas are unavailable.
    """

    def __init__(self, timestamp: float, transcript: str):
        super().__init__(timestamp)
        self.transcript = transcript

    def __repr__(self) -> str:
        return f"ContentDoneEvent(ts={self.timestamp:.3f}, transcript={self.transcript!r})"


class AudioOutputEvent(StreamEvent):
    """Model produced an audio output chunk (used for duration tracking)."""

    def __init__(self, timestamp: float, audio_bytes: Optional[bytes] = None):
        super().__init__(timestamp)
        self.audio_bytes = audio_bytes

    def __repr__(self) -> str:
        size = len(self.audio_bytes) if self.audio_bytes else 0
        return f"AudioOutputEvent(ts={self.timestamp:.3f}, bytes={size})"


class ErrorEvent(StreamEvent):
    """An error occurred during the session."""

    def __init__(self, timestamp: float, message: str):
        super().__init__(timestamp)
        self.message = message

    def __repr__(self) -> str:
        return f"ErrorEvent(ts={self.timestamp:.3f}, message={self.message!r})"


class SessionEndEvent(StreamEvent):
    """Model finished responding (analogous to ``response.done``)."""

    pass


EventType = Union[
    SessionReadyEvent,
    ContentTokenEvent,
    ContentDoneEvent,
    AudioOutputEvent,
    ErrorEvent,
    SessionEndEvent,
]


# ── abstract client ────────────────────────────────────────────────


class StreamingClient(ABC):
    """Abstract base for a streaming speech-to-text API backend.

    Subclasses implement the five lifecycle methods.  They must emit
    ``StreamEvent`` instances through ``iter_events``, and must support
    concurrent access: ``send_audio`` / ``end_audio`` run in the application
    thread while ``iter_events`` is consumed (potentially on the same or a
    separate thread).

    The canonical lifecycle is::

        client = ConcreteClient()
        client.connect(config)
        for chunk in audio_chunks:
            client.send_audio(chunk, chunk_id)
        client.end_audio()
        for event in client.iter_events():
            process(event)
        client.disconnect()
    """

    # ── lifecycle ──────────────────────────────────────────────

    @abstractmethod
    def connect(self, config: SessionConfig) -> None:
        """Open the connection and initialise the session.

        Must emit at least one ``SessionReadyEvent`` once the session
        is established.
        """
        ...

    @abstractmethod
    def send_audio(self, audio_bytes: bytes, chunk_id: int) -> None:
        """Push one chunk of audio data to the model.

        Parameters
        ----------
        audio_bytes : bytes
            Raw PCM audio (int16, mono, target sample rate).
        chunk_id : int
            Sequential chunk index (0-based), used for audio-aligned
            timing correction.
        """
        ...

    @abstractmethod
    def end_audio(self) -> None:
        """Signal that no more audio will be sent (commit + trigger response)."""
        ...

    @abstractmethod
    def iter_events(self) -> Iterator[StreamEvent]:
        """Yield ``StreamEvent`` items as they arrive from the API.

        This is expected to block until the session ends or an error
        occurs.  Implementations may buffer events internally and drain
        them here, or yield directly from a callback.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release resources."""
        ...
