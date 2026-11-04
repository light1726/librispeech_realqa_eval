"""Reads benchmark JSONL and yields DatasetSample objects."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any

from realqa.sample import DatasetSample, Emission


class DatasetReader:
    """Iterable over a LibriSpeech realtime-QA benchmark JSONL file.

    Each JSONL line is expected to have::

        {
          "utterance_id": str,
          "split": str,
          "audio_path": str,
          "task_type": str,       # e.g. "lexical_detection", "semantic_detection"
          "query": str,
          "prompt": str,
          "labels": [[t_start, t_end, text], ...]
        }

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the benchmark JSONL file.
    task_type : str or None
        If provided, only yield samples matching this task_type.
    limit : int or None
        If set, yield at most *limit* samples.
    seed : int or None
        Seed for deterministic shuffling before limiting.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        task_type: Optional[str] = None,
        limit: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self._path = Path(jsonl_path)
        self._task_type = task_type
        self._limit = limit
        self._seed = seed

    def _load_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __iter__(self) -> Iterator[DatasetSample]:
        rows = self._load_rows()
        if self._task_type is not None:
            rows = [r for r in rows if r.get("task_type") == self._task_type]
        if self._seed is not None:
            rng = random.Random(self._seed)
            rng.shuffle(rows)
        if self._limit is not None:
            rows = rows[: self._limit]
        for row in rows:
            yield _row_to_sample(row)

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _row_to_sample(row: Dict[str, Any]) -> DatasetSample:
    required = ("utterance_id", "audio_path", "prompt")
    missing = [k for k in required if k not in row]
    if missing:
        raise KeyError(f"missing required keys: {missing} in row: {row}")

    labels_raw = row.get("labels", [])
    emissions = [
        Emission(text=str(l[2]), t_start=float(l[0]), t_end=float(l[1]))
        for l in labels_raw
    ]
    return DatasetSample(
        utterance_id=row["utterance_id"],
        audio_path=row["audio_path"],
        prompt=row["prompt"],
        task_type=row.get("task_type", ""),
        query=row.get("query", ""),
        emissions=emissions,
    )
