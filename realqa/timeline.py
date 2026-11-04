"""Frame-grid timeline builder.

Converts a stream of ``StreamEvent`` items into a token timeline at a
fixed frame rate, suitable for feeding into ``StreamingMetrics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from realqa.base import (
    ContentTokenEvent,
    ContentDoneEvent,
    StreamEvent,
    TimelineResult,
)


@dataclass
class ContentRun:
    """Intermediate representation of one content burst."""

    start_idx: int       # first frame index (inclusive)
    end_idx: int         # exclusive end / EOS boundary
    tokens: List[str]    # delta tokens in this burst (in order)
    text: str            # decoded text (joined tokens)
    t_start: float       # start time in seconds
    t_end: float         # end time in seconds
    event_times: List[Tuple[float, str]] = field(default_factory=list)
    # (timestamp_sec, token) for each delta, used for frame placement


class TimelineBuilder:
    """Build a token timeline from API events at a fixed frame rate.

    Each content delta is placed at the frame corresponding to its
    wall-clock timestamp.  Unfilled frames stay ``silence_token``.  If
    responses arrive after audio ends, the grid is extended with
    additional silence frames so timestamps are preserved.  Deltas that
    land on the same frame are concatenated (e.g. ``\"wes\"+\"ley\"``).

    Parameters
    ----------
    chunks_per_second : float
        Grid resolution in Hz.  For API inference, use ``1000 / chunk_ms``
        (e.g. 5.0 for 200 ms chunks).
    silence_token : str
        Token placed in silence positions.
    eos_token : str
        Optional token marking the end of each content burst.
        Empty string (default) means no EOS marker is inserted.
    """

    def __init__(
        self,
        chunks_per_second: float = 5.0,
        silence_token: str = "<silence>",
        eos_token: str = "",
    ):
        self.chunks_per_second = chunks_per_second
        self.silence_token = silence_token
        self.eos_token = eos_token

    def build(
        self,
        events: List[StreamEvent],
        audio_duration_s: float,
        prefix_len: int = 0,
    ) -> TimelineResult:
        """Convert events + audio duration into a token grid."""
        bursts = self._extract_bursts(events)

        max_t = audio_duration_s
        for burst in bursts:
            max_t = max(max_t, burst.t_start, burst.t_end)
            for ts, _ in burst.event_times:
                max_t = max(max_t, ts)

        n_frames = max(1, math.ceil(max_t * self.chunks_per_second))
        for burst in bursts:
            for ts, _ in burst.event_times:
                n_frames = max(n_frames, self._raw_frame(ts) + 1)
            if self.eos_token and burst.event_times:
                n_frames = max(n_frames, self._raw_frame(burst.t_end) + 2)

        tokens = [self.silence_token] * n_frames

        for burst in bursts:
            first_idx: Optional[int] = None
            last_idx: Optional[int] = None
            for ts, tok in burst.event_times:
                idx = min(self._raw_frame(ts), n_frames - 1)
                if tokens[idx] == self.silence_token:
                    tokens[idx] = tok
                else:
                    tokens[idx] = tokens[idx] + tok
                if first_idx is None:
                    first_idx = idx
                last_idx = idx

            if first_idx is None:
                continue
            burst.start_idx = first_idx
            burst.end_idx = last_idx + 1  # type: ignore[operator]

            if self.eos_token:
                eos_idx = burst.end_idx
                if eos_idx < n_frames and tokens[eos_idx] == self.silence_token:
                    tokens[eos_idx] = self.eos_token
                    burst.end_idx = eos_idx + 1

        content_runs = [
            {
                "text": b.text,
                "t_start": b.t_start,
                "t_end": b.t_end,
                "start_group": b.start_idx,
                "end_group": b.end_idx,
            }
            for b in bursts
            if b.event_times
        ]

        return TimelineResult(
            tokens=tokens,
            content_runs=content_runs,
            chunks_per_second=self.chunks_per_second,
            prefix_len=prefix_len,
        )

    # ── internal helpers ────────────────────────────────────────

    def _extract_bursts(self, events: List[StreamEvent]) -> List[ContentRun]:
        bursts: List[ContentRun] = []
        current_group: List[ContentTokenEvent] = []

        for ev in events:
            if isinstance(ev, ContentTokenEvent):
                current_group.append(ev)
            elif isinstance(ev, ContentDoneEvent):
                if current_group:
                    bursts.append(self._tokens_to_run(current_group, end_time=ev.timestamp))
                    current_group = []
                elif ev.transcript:
                    bursts.append(self._done_to_run(ev))

        if current_group:
            bursts.append(self._tokens_to_run(current_group))

        return bursts

    def _done_to_run(self, done: ContentDoneEvent) -> ContentRun:
        ts = done.timestamp
        text = done.transcript.strip()
        return ContentRun(
            start_idx=0,
            end_idx=0,
            tokens=[text] if text else [],
            text=text,
            t_start=ts,
            t_end=ts,
            event_times=[(ts, text)] if text else [],
        )

    def _tokens_to_run(
        self,
        token_events: List[ContentTokenEvent],
        end_time: Optional[float] = None,
    ) -> ContentRun:
        start_time = token_events[0].timestamp
        boundary = end_time if end_time is not None else token_events[-1].timestamp
        text = "".join(t.token for t in token_events)
        return ContentRun(
            start_idx=0,
            end_idx=0,
            tokens=[t.token for t in token_events],
            text=text,
            t_start=start_time,
            t_end=boundary,
            event_times=[(t.timestamp, t.token) for t in token_events],
        )

    def _raw_frame(self, seconds: float) -> int:
        """Convert seconds to a non-negative frame index (unclamped upper)."""
        return max(0, int(seconds * self.chunks_per_second + 1e-9))
