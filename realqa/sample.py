"""Data types for dataset records used across the inference + eval pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Union


@dataclass
class Emission:
    """A single ground-truth word with audio-aligned timing."""

    text: str
    t_start: float
    t_end: float


def make_sample_id(utterance_id: str, task_type: str, query: str) -> str:
    """Stable unique id for one benchmark row (utt × task × query)."""
    return f"{utterance_id}::{task_type}::{query}"


def emissions_fingerprint(
    utterance_id: str,
    emissions: Sequence[Union[Emission, dict]],
) -> str:
    """Fingerprint used to match legacy preds that only stored ``utterance_id``.

    Old prediction lines used ``sample_id == utterance_id``, so resume cannot
    tell tasks apart by id alone; matching GT emissions recovers the row.
    """
    parts: List[Tuple[str, float, float]] = []
    for em in emissions:
        if isinstance(em, dict):
            parts.append(
                (
                    str(em.get("text", "")),
                    round(float(em["t_start"]), 4),
                    round(float(em["t_end"]), 4),
                )
            )
        else:
            parts.append((em.text, round(em.t_start, 4), round(em.t_end, 4)))
    return utterance_id + "||" + json.dumps(parts, ensure_ascii=False)


@dataclass
class DatasetSample:
    """One benchmark entry: audio file + prompt + ground-truth emissions."""

    utterance_id: str
    audio_path: str
    prompt: str
    task_type: str
    query: str
    emissions: List[Emission] = field(default_factory=list)

    @property
    def sample_id(self) -> str:
        """Unique across multi-task rows that share the same audio clip."""
        return make_sample_id(self.utterance_id, self.task_type, self.query)

    @property
    def fingerprint(self) -> str:
        return emissions_fingerprint(self.utterance_id, self.emissions)
