#!/usr/bin/env python3
"""Evaluate RealQA / realtime-QA predictions from a JSONL file.

Each line (``ref`` is optional — ``emissions`` + ``timeline_len`` suffice)::

    {
      "sample_id": "utt-001",
      "pred": ["<silence>", "...", "stew", "<|im_end|>", ...],
      "timeline_len": 116,
      "prefix_len": 33,
      "chunks_per_second": 6.25,
      "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}]
    }

``pred`` / ``ref`` are token-string lists, or strings with the special
tokens embedded. Acceptance windows are always during_word (g=0) /
+1c (g=C) / +2c (g=2C), with C = 1/chunks_per_second per row unless
--grace-window-sec pins it.

CLI knobs (defaults, overridable per line)::

    --silence-token        e.g. <silence>
    --eos-token            e.g. <|im_end|>  (repeatable / comma-separated)
    --chunks-per-second    e.g. 6.25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from realqa.metrics import StreamingMetrics, TimelineItem, json_safe
from realqa.post_audio_metrics import PostAudioMetrics
from realqa.tokenize import (
    VocabTokenizer,
    normalize_sequence,
)


def _parse_eos_tokens(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_pred_runs(raw, sample_id: str) -> Optional[List[Dict[str, Any]]]:
    """Normalize the optional ``pred_runs`` field (event-level predictions).

    Entries need ``text`` and ``t_start`` (seconds); ``t_end`` optional.
    Returns ``None`` when the field is absent (grid fallback).
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{sample_id}: pred_runs must be a list, got {type(raw).__name__}")
    units = []
    for u in raw:
        if not isinstance(u, dict) or u.get("text") is None or u.get("t_start") is None:
            raise ValueError(
                f"{sample_id}: pred_runs entries need text and t_start: {u!r}"
            )
        units.append({
            "text": str(u["text"]),
            "t_start": float(u["t_start"]),
            "t_end": float(u["t_end"]) if u.get("t_end") is not None else None,
        })
    return units


def _resolve_emissions(
    raw: Optional[list],
    *,
    chunks_per_second: float,
    prefix_len: int,
    ref_tokens: Optional[List[str]],
    silence_token: str,
    tokenizer: VocabTokenizer,
) -> List[Dict[str, Any]]:
    """Normalize emissions; optionally derive from ref non-wait runs."""
    if raw:
        emissions = []
        for em in raw:
            text = em.get("text")
            t_start = em.get("t_start")
            t_end = em.get("t_end")
            sg = em.get("start_group")
            eg = em.get("end_group")
            if t_end is None and eg is not None:
                t_end = float(eg) / chunks_per_second
            if t_start is None and sg is not None:
                t_start = float(sg) / chunks_per_second
            if text is None:
                raise ValueError(f"emission missing text: {em}")
            if t_end is None:
                raise ValueError(
                    f"emission needs t_end or end_group: {em}"
                )
            emissions.append({
                "text": text,
                "t_start": t_start,
                "t_end": t_end,
                "start_group": sg,
                "end_group": eg,
            })
        return emissions

    if ref_tokens is None:
        raise ValueError(
            "emissions are required when 'ref' field is not provided"
        )

    # Derive from reference timeline: each non-wait run → one emission.
    # Timing falls back to group-space relative to prefix_len.
    silence_id = tokenizer.silence_token_id
    ids = tokenizer.encode_tokens(ref_tokens)
    runs = StreamingMetrics._segment_nonwait_runs(ids, silence_id)
    emissions = []
    for (s, e) in runs:
        text = tokenizer.decode(ids[s:e], skip_special_tokens=True)
        start_group = s - prefix_len
        end_group = e - 1 - prefix_len
        emissions.append({
            "text": text,
            "t_start": start_group / chunks_per_second,
            "t_end": end_group / chunks_per_second,
            "start_group": start_group,
            "end_group": end_group,
            "derived": True,
        })
    return emissions


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def build_items(
    rows: List[Dict[str, Any]],
    *,
    silence_token: str,
    eos_tokens: List[str],
    default_chunks_per_second: float,
) -> tuple[List[TimelineItem], List[TimelineItem], VocabTokenizer]:
    # First pass: collect all tokens to build vocab.
    all_token_seqs: List[tuple] = []
    parsed_rows = []
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is None:
            raise ValueError(f"missing sample_id in row: {row}")
        if "pred" not in row:
            raise ValueError(f"{sample_id}: need pred field")
        row_wait = row.get("silence_token", silence_token)
        row_eos = row.get("eos_tokens", eos_tokens)
        if isinstance(row_eos, str):
            row_eos = _parse_eos_tokens([row_eos])
        pred_toks = normalize_sequence(
            row["pred"], silence_token=row_wait, eos_tokens=row_eos,
        )

        timeline_len = int(row.get("timeline_len", 0))
        ref_present = "ref" in row
        if ref_present:
            ref_toks = normalize_sequence(
                row["ref"], silence_token=row_wait, eos_tokens=row_eos,
            )
            if not timeline_len:
                timeline_len = len(ref_toks)
        else:
            # No ref — derive timeline_len from pred length or explicit field
            if not timeline_len:
                timeline_len = len(pred_toks)
            # Build a dummy ref of silence tokens so vocab stays consistent
            ref_toks = [row_wait] * timeline_len

        all_token_seqs.append((pred_toks, ref_toks))
        parsed_rows.append((row, sample_id, row_wait, row_eos, pred_toks, ref_toks, timeline_len, ref_present))

    tokenizer = VocabTokenizer(silence_token=silence_token, eos_tokens=eos_tokens)
    for pred_toks, ref_toks in all_token_seqs:
        for t in pred_toks:
            tokenizer.add_token(t)
        for t in ref_toks:
            tokenizer.add_token(t)

    preds: List[TimelineItem] = []
    refs: List[TimelineItem] = []
    for row, sample_id, row_wait, row_eos, pred_toks, ref_toks, timeline_len, ref_present in parsed_rows:
        # Per-row silence token must match the shared vocab wait for segmentation.
        if row_wait != silence_token:
            raise ValueError(
                f"{sample_id}: per-line silence_token={row_wait!r} differs from "
                f"CLI --silence-token={silence_token!r}; use one silence token per eval run"
            )
        chunks_per_second = float(row.get("chunks_per_second", default_chunks_per_second))
        # Nonzero only for silence-token model timelines whose grid starts
        # with prompt groups before audio time zero; API timelines use 0.
        prefix_len = int(row.get("prefix_len", 0) or 0)

        pred_ids = tokenizer.encode_tokens(pred_toks)
        ref_ids = tokenizer.encode_tokens(ref_toks)
        L = timeline_len

        emissions = _resolve_emissions(
            row.get("emissions"),
            chunks_per_second=chunks_per_second,
            prefix_len=prefix_len,
            ref_tokens=ref_toks if ref_present else None,
            silence_token=silence_token,
            tokenizer=tokenizer,
        )
        for em in emissions:
            t_s, t_e = em.get("t_start"), em.get("t_end")
            if (
                t_s is None or t_e is None
                or not (math.isfinite(t_s) and math.isfinite(t_e))
                or t_s > t_e
            ):
                raise ValueError(
                    f"{sample_id}: emission needs finite t_start <= t_end, "
                    f"got t_start={t_s!r}, t_end={t_e!r} (text={em.get('text')!r})"
                )

        preds.append(TimelineItem(
            sample_id=sample_id,
            pred_full_text_tokens=pred_ids,
            pred_full_text="".join(pred_toks),
            pred_units=_parse_pred_runs(row.get("pred_runs"), sample_id),
            silence_token_id=tokenizer.silence_token_id,
        ))
        ref_kwargs: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt": row.get("prompt"),
            "ref_emissions": emissions,
            "prefix_len": prefix_len,
            "timeline_len": L,
            "chunks_per_second": chunks_per_second,
            "silence_token_id": tokenizer.silence_token_id,
        }
        if ref_present:
            ref_kwargs["ref_full_text_tokens"] = ref_ids
            ref_kwargs["ref_full_text"] = "".join(ref_toks)
        refs.append(TimelineItem(**ref_kwargs))
    return preds, refs, tokenizer


def write_summary_csv(path: Path, metrics: Dict[str, Any]) -> None:
    # Flat single-row CSV of dataset-level scalars. NaN becomes an empty
    # cell so the column set is stable across runs.
    scalar = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            scalar[k] = "" if (isinstance(v, float) and v != v) else v
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar.keys()))
        writer.writeheader()
        writer.writerow(scalar)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone RealQA / realtime-QA JSONL evaluation",
    )
    parser.add_argument("jsonl", type=Path, help="Input predictions JSONL")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all_results.json, summary.csv, per-sample dumps",
    )
    parser.add_argument(
        "--silence-token", type=str, default="<silence>",
        help="Model silence special token string",
    )
    parser.add_argument(
        "--eos-token", action="append", default=None,
        help="EOS / boundary special token(s). Repeatable or comma-separated. "
             "Skipped when decoding content text.",
    )
    parser.add_argument(
        "--chunks-per-second", type=float, default=6.25,
        help="Audio token / group frame rate in Hz (default: 6.25)",
    )
    parser.add_argument(
        "--metric-mode",
        type=str,
        choices=("streaming", "post_audio"),
        default="streaming",
        help="streaming: grace-window StreamingMetrics; "
             "post_audio: ordered exclusive alignment + onset delay",
    )
    parser.add_argument(
        "--early-eps-sec",
        type=float,
        default=0.05,
        help="post_audio only: reject pred onset earlier than "
             "ref_t_start - early_eps_sec (default: 0.05)",
    )
    parser.add_argument(
        "--grace-window-sec", type=float, default=None,
        help="Evaluation-only: chunk duration C for the acceptance windows "
             "during_word(0) / +1c(C) / +2c(2C). When omitted, C is derived "
             "per row as 1/chunks_per_second (model timelines). Pass this "
             "for API predictions so the protocol does not depend on the "
             "transport chunk size (e.g. 0.16 for the 160 ms protocol).",
    )
    parser.add_argument(
        "--onset-source",
        type=str,
        choices=("auto", "events", "grid"),
        default="auto",
        help="Prediction unit/timing source: 'events' uses the per-run "
             "exact onsets from the pred_runs field (one unit per model "
             "burst); 'grid' re-derives units from the token grid "
             "(native timing for silence-token models); 'auto' uses events "
             "per row when pred_runs is present and grid otherwise "
             "(default).",
    )
    args = parser.parse_args(argv)

    eos_tokens = _parse_eos_tokens(args.eos_token) or ["<|im_end|>"]

    rows = load_jsonl(args.jsonl)
    if not rows:
        print("No samples in JSONL", file=sys.stderr)
        return 1

    preds, refs, tokenizer = build_items(
        rows,
        silence_token=args.silence_token,
        eos_tokens=eos_tokens,
        default_chunks_per_second=args.chunks_per_second,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.metric_mode == "post_audio":
        metric = PostAudioMetrics(
            early_eps_sec=args.early_eps_sec,
            onset_source=args.onset_source,
        )
        metrics_key = "post_audio_metrics"
        task_type = "POST_AUDIO"
        headline_keys = [
            "emission_f1_fuzzy",
            "emission_recall_fuzzy",
            "emission_precision_fuzzy",
            "fdr",
            "delay_mean",
            "n_samples",
        ]
    else:
        metric = StreamingMetrics(
            onset_source=args.onset_source,
            grace_window_sec=args.grace_window_sec,
        )
        metrics_key = "streaming_metrics"
        task_type = "STREAMING"

    sample_results = metric.compute_batch(
        preds, refs, output_dir=args.output_dir, tokenizer=tokenizer,
    )
    dataset_metrics = metric.compute_dataset_level_metrics(sample_results)

    all_results = {
        "dataset_name": args.jsonl.stem,
        "task_type": task_type,
        "num_samples": len(sample_results),
        "config": {
            "metric_mode": args.metric_mode,
            "onset_source": args.onset_source,
            "grace_window_sec": args.grace_window_sec,
            "silence_token": args.silence_token,
            "eos_tokens": eos_tokens,
            "chunks_per_second": args.chunks_per_second,
            "early_eps_sec": args.early_eps_sec if args.metric_mode == "post_audio" else None,
        },
        "dataset_level_metrics": {metrics_key: dataset_metrics},
        "sample_level_metrics": {metrics_key: sample_results},
    }
    out_json = args.output_dir / "all_results.json"
    with out_json.open("w") as f:
        json.dump(json_safe(all_results), f, indent=2)

    write_summary_csv(args.output_dir / "summary.csv", dataset_metrics)

    print(f"Wrote {out_json} ({len(sample_results)} samples, mode={args.metric_mode})")
    if args.metric_mode == "post_audio":
        for k in headline_keys:
            if k in dataset_metrics:
                v = dataset_metrics[k]
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        # One line per grace window: every field on a line comes from the
        # same window / match set, so printed rows cannot mix windows.
        print(f"  n_samples: {dataset_metrics.get('n_samples')}")
        for wname in metric.WINDOW_NAMES:
            tp = dataset_metrics.get(f"total_tp_fuzzy_{wname}", 0)
            fa = dataset_metrics.get(f"total_false_alarm_fuzzy_{wname}", 0)
            fdr_w = (fa / (fa + tp)) if (fa + tp) > 0 else 0.0
            f1 = dataset_metrics.get(f"emission_f1_fuzzy_{wname}")
            prec = dataset_metrics.get(f"emission_precision_fuzzy_{wname}")
            rec = dataset_metrics.get(f"emission_recall_fuzzy_{wname}")
            delay = dataset_metrics.get(f"delay_mean_fuzzy_{wname}")
            delay_s = "n/a" if delay is None or delay != delay else f"{delay:.4f}"
            print(
                f"  [{wname}] f1_fuzzy={f1:.4f} precision={prec:.4f} "
                f"recall={rec:.4f} fdr={fdr_w:.4f} delay_mean={delay_s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
