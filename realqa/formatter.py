"""Eval JSONL formatter.

Merges a ``DatasetSample`` (reference ground-truth) with a
``TimelineResult`` (prediction token grid) into a single JSONL line
compatible with ``eval_jsonl.py``.

The reference side only carries ``timeline_len`` and explicit
``emissions`` — the ``ref`` token grid is no longer needed because the
metrics engine matches against ``emissions`` directly.

``pred_runs`` records one entry per model burst with the exact
first-delta timestamp, so evaluation scores event-level units without
frame quantization or grid-contiguity artifacts (the token grid is kept
for visualization and as the grid fallback).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any

from realqa.sample import DatasetSample
from realqa.base import TimelineResult


@dataclass
class EvalFormatter:
    """Produces one eval-ready JSONL line from a sample + prediction.

    Parameters
    ----------
    silence_token : str
        Silence token placed in prediction reference positions.
    eos_token : str
        Optional end-of-segment boundary token (unused in formatting;
        kept for CLI symmetry with TimelineBuilder).
    """

    silence_token: str = "<silence>"
    eos_token: str = ""

    def format_line(
        self,
        sample: DatasetSample,
        timeline: TimelineResult,
    ) -> Dict[str, Any]:
        """Build a dict suitable for writing as one JSONL line."""

        return {
            "sample_id": sample.sample_id,
            "utterance_id": sample.utterance_id,
            "task_type": sample.task_type,
            "query": sample.query,
            "pred": timeline.tokens,
            "pred_runs": [
                {"text": r["text"], "t_start": r["t_start"], "t_end": r["t_end"]}
                for r in timeline.content_runs
            ],
            "timeline_len": len(timeline.tokens),
            "prefix_len": timeline.prefix_len,
            "chunks_per_second": timeline.chunks_per_second,
            "emissions": [
                {"text": em.text, "t_start": em.t_start, "t_end": em.t_end}
                for em in sample.emissions
            ],
        }

    def format_line_json(self, sample: DatasetSample, timeline: TimelineResult) -> str:
        """Return a single JSON string (with trailing newline) for the JSONL."""
        return json.dumps(self.format_line(sample, timeline), ensure_ascii=False) + "\n"
