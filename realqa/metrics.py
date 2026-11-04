"""Standalone RealQA streaming evaluation metrics.

Vendored/adapted from duplex_spoken_dialogue.
Timing is computed in seconds via
``onset_sec = (token_pos - prefix_len) / chunks_per_second``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class TimelineItem:
    """One predicted or reference silence-token timeline.

    ``ref_full_text_tokens`` / ``ref_full_text`` may be ``None`` when
    the reference side is expressed purely through ``ref_emissions`` and
    ``timeline_len``.
    """

    sample_id: str = None
    prompt: Optional[str] = None
    pred_full_text_tokens: torch.Tensor = None
    ref_full_text_tokens: Optional[torch.Tensor] = None
    pred_full_text: Optional[str] = None
    ref_full_text: Optional[str] = None
    ref_emissions: Optional[list] = None  # [{text, t_start, t_end, ...}]
    # Event-level predictions: [{text, t_start, t_end}] with the exact
    # first-delta timestamp per burst (JSONL ``pred_runs``).  When present
    # (and onset policy allows) these replace grid-derived units so burst
    # boundaries and onsets are not quantized to frame resolution.
    pred_units: Optional[list] = None
    prefix_len: Optional[int] = None
    timeline_len: Optional[int] = None
    chunks_per_second: Optional[float] = None
    silence_token_id: Optional[int] = None


ONSET_SOURCES: Tuple[str, ...] = ("auto", "events", "grid")


def json_safe(obj):
    """Recursively convert NaN floats to None for strict-JSON output."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def resolve_pred_units(
    pred: TimelineItem,
    *,
    pred_seq: torch.Tensor,
    silence_id: int,
    prefix_len: int,
    chunks_per_second: float,
    tokenizer=None,
    onset_source: str = "auto",
    sample_id: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    """Return ``(units, source)`` describing the predicted answer units.

    Each unit is ``{"text", "compact", "onset_sec"}`` where ``text`` /
    ``compact`` are normalized.  Two sources:

    - ``events``: one unit per recorded burst (``pred_runs``), onset =
      the exact first-delta timestamp.  Requires ``pred.pred_units``.
    - ``grid``: one unit per contiguous non-wait run on the token grid,
      onset = ``(frame_index - prefix_len) / chunks_per_second``.  This is the
      native (causal, chunk-resolution) timing for silence-token models and
      a floor-quantization of true timestamps for API runs.

    ``onset_source`` policy: ``auto`` uses events per row when available
    (grid otherwise); ``events`` demands events for every row; ``grid``
    always re-derives from the grid.
    """
    if onset_source not in ONSET_SOURCES:
        raise ValueError(
            f"onset_source must be one of {ONSET_SOURCES}, got {onset_source!r}"
        )
    events = pred.pred_units
    if events is not None and onset_source != "grid":
        units = []
        for u in events:
            txt = u.get("text") or ""
            units.append({
                "text": StreamingMetrics._norm_text(txt),
                "compact": StreamingMetrics._norm_compact(txt),
                "onset_sec": float(u["t_start"]),
            })
        return units, "events"
    if events is None and onset_source == "events":
        raise ValueError(
            f"{sample_id}: onset_source='events' requires a pred_runs field"
        )
    units = []
    for (s, e) in StreamingMetrics._segment_nonwait_runs(pred_seq, silence_id):
        txt = (
            tokenizer.decode(pred_seq[s:e], skip_special_tokens=True)
            if tokenizer is not None else ""
        )
        units.append({
            "text": StreamingMetrics._norm_text(txt),
            "compact": StreamingMetrics._norm_compact(txt),
            "onset_sec": (s - prefix_len) / chunks_per_second,
        })
    return units, "grid"


class StreamingMetrics:
    """Emission matching over silence-token timelines (fuzzy/exact × grace windows).

    For each reference emission, each grace ``g``, and each match mode we run an
    independent assignment on ``[t_start, t_end + g]``. In-window matching pred
    runs are **merged** into one TP unit; ``FA = pred_runs - matched_preds``.

    Dump ``match_type`` is labeled from the **merged** unit (compact-equal →
    ``exact``, else ``fuzzy``). Over-emission is fuzzy even when each run alone
    would be exact — e.g. ref ``"hi"`` with preds ``"hi"`` + ``"hi"`` merges to
    ``"hihi"``, so ``match_type="fuzzy"``.

    Headline ``fdr = FA / (FA + TP)`` (broadest fuzzy); dataset-level ``fdr`` and
    ``exact_match_rate`` are **micro-averaged**. Task type is not inferred from
    the prompt — callers already evaluate separate subsets.
    """

    # Acceptance-window protocol, always on: during_word (g = 0),
    # +1c (g = C), +2c (g = 2C), C = 1/chunks_per_second per row unless
    # grace_window_sec pins it.
    WINDOW_NAMES: Tuple[str, ...] = ("during_word", "plus1c", "plus2c")
    MATCH_KINDS: Tuple[str, ...] = ("fuzzy", "exact")

    def __init__(
        self,
        onset_source: str = "auto",
        grace_window_sec: Optional[float] = None,
    ):
        if onset_source not in ONSET_SOURCES:
            raise ValueError(
                f"onset_source must be one of {ONSET_SOURCES}, got {onset_source!r}"
            )
        if grace_window_sec is not None and float(grace_window_sec) <= 0:
            raise ValueError(f"grace_window_sec must be positive, got {grace_window_sec}")
        self.onset_source = onset_source
        # Protocol chunk duration C for the acceptance windows. When None it
        # is derived per row (C = 1 / chunks_per_second) — correct for model
        # timelines whose chunk rate IS the model's. Set it explicitly to pin
        # the protocol regardless of the (transport-chosen) timeline rate,
        # e.g. for API predictions.
        self.grace_window_sec = (
            float(grace_window_sec) if grace_window_sec is not None else None
        )

    def windows_for(self, chunks_per_second: float) -> List[Tuple[str, float]]:
        """Acceptance windows for one row: g = 0, C, 2C.

        C = grace_window_sec when set, else 1 / chunks_per_second.
        """
        c = (
            self.grace_window_sec
            if self.grace_window_sec is not None
            else 1.0 / float(chunks_per_second)
        )
        return [("during_word", 0.0), ("plus1c", c), ("plus2c", 2.0 * c)]

    @staticmethod
    def _segment_nonwait_runs(tokens, silence_id: int) -> List[Tuple[int, int]]:
        runs = []
        tok = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
        i, n = 0, len(tok)
        while i < n:
            if tok[i] == silence_id:
                i += 1
                continue
            j = i
            while j < n and tok[j] != silence_id:
                j += 1
            runs.append((i, j))
            i = j
        return runs

    @staticmethod
    def _norm_text(s: str) -> str:
        return " ".join((s or "").lower().strip().split())

    @staticmethod
    def _norm_compact(s: str) -> str:
        return "".join((s or "").lower().split())

    # Cardinal word ↔ digit (and vice versa) for answer matching.
    _WORD_TO_DIGIT = {
        "zero": "0", "oh": "0",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
        "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
        "hundred": "100",
    }
    _DIGIT_TO_WORD = {v: k for k, v in _WORD_TO_DIGIT.items() if k != "oh"}

    @classmethod
    def _alnum_compact(cls, s: str) -> str:
        """Lowercase compact form with non-alphanumerics stripped."""
        return "".join(ch for ch in cls._norm_compact(s) if ch.isalnum())

    @classmethod
    def _number_forms(cls, compact: str) -> set:
        """Compact string plus digit/word number aliases (whole-token).

        ``\"one\"`` ↔ ``\"1\"``, ``\"two\"`` ↔ ``\"2\"``, etc.  Multi-token
        compact strings are left unchanged aside from alnum stripping.
        """
        base = cls._alnum_compact(compact)
        if not base:
            return set()
        forms = {base}
        if base in cls._WORD_TO_DIGIT:
            forms.add(cls._WORD_TO_DIGIT[base])
        if base in cls._DIGIT_TO_WORD:
            forms.add(cls._DIGIT_TO_WORD[base])
        # Also try mapping whitespace-tokenized original if compact had no spaces
        # (already single token). For spaced text passed via compact wrongly,
        # nothing else to do.
        return forms

    @classmethod
    def _content_matches(cls, ref_compact: str, pred_compact: str) -> Tuple[bool, bool]:
        """Return (matches, is_exact). Exact = form-equal; fuzzy = ref in pred.

        Number words and digits are treated as equivalent (``one`` ↔ ``1``).
        """
        if not cls._alnum_compact(ref_compact):
            return False, False
        ref_forms = cls._number_forms(ref_compact)
        pred_forms = cls._number_forms(pred_compact)
        if not ref_forms:
            return False, False
        # Exact: any shared canonical form (incl. number aliases).
        if ref_forms & pred_forms:
            return True, True
        # Fuzzy: any ref form is a substring of any pred form.
        for rf in ref_forms:
            for pf in pred_forms:
                if rf and rf in pf:
                    return True, False
        return False, False

    @classmethod
    def _exact_matches(cls, ref_compact: str, pred_compact: str) -> Tuple[bool, bool]:
        if not cls._alnum_compact(ref_compact):
            return False, False
        ref_forms = cls._number_forms(ref_compact)
        pred_forms = cls._number_forms(pred_compact)
        if not ref_forms:
            return False, False
        eq = bool(ref_forms & pred_forms)
        return eq, eq

    def _match_at_grace(
        self,
        grace: float,
        match_fn,
        ref_emissions,
        pred_runs,
        pred_onsets_sec,
        pred_texts,
        pred_compacts,
        ref_compacts,
        ref_t_start,
        ref_t_end,
        kind: str = "fuzzy",
    ):
        """Independent assignment for one grace and one matcher.

        Dump ``match_type`` uses the **merged** compact string vs the ref.
        Repeated in-window emissions (``"hi"`` + ``"hi"`` → ``"hihi"``) are
        labeled ``fuzzy``: over-emission is not an exact match.
        """
        matched_preds: set = set()
        run_fate = ["fa"] * len(pred_runs)
        run_meta: List[Optional[str]] = [None] * len(pred_runs)
        emission_results = []
        tp = 0
        for i, em in enumerate(ref_emissions):
            rc = ref_compacts[i]
            t_s = ref_t_start[i]
            t_e = ref_t_end[i]
            lo = t_s if t_s is not None else float("-inf")
            hi = (t_e + grace) if t_e is not None else float("inf")
            in_win_matched = []
            for k in range(len(pred_runs)):
                onset_k = pred_onsets_sec[k]
                if onset_k < lo - 1e-9 or onset_k > hi + 1e-9:
                    continue
                m, _ = match_fn(rc, pred_compacts[k])
                if m:
                    in_win_matched.append(k)
            hit = False
            if in_win_matched and t_e is not None:
                combined_compact = "".join(pred_compacts[k] for k in in_win_matched)
                combined_text = " ".join(pred_texts[k] for k in in_win_matched)
                combined_onset = min(pred_onsets_sec[k] for k in in_win_matched)
                # Label from merged unit vs ref (independent of assignment kind).
                # Over-emission (hi+hi→hihi) is fuzzy even if each run alone was exact.
                _, is_exact = self._content_matches(rc, combined_compact)
                match_type = "exact" if is_exact else "fuzzy"
                if kind != "exact" or is_exact:
                    # The exact tier credits the emission only when the merged
                    # unit itself is form-equal (summary == detail label).
                    for k in in_win_matched:
                        matched_preds.add(k)
                        run_fate[k] = "tp"
                        run_meta[k] = match_type
                    tp += 1
                    hit = True
                    delay = (
                        (combined_onset - float(t_s))
                        if t_s is not None
                        else None
                    )
                    emission_results.append({
                        "text": em["text"],
                        "t_start": t_s, "t_end": t_e,
                        "hit": True, "missed": False,
                        "match_type": match_type,
                        "exact_match": is_exact,
                        "pred_text": combined_text,
                        "pred_onset_sec": combined_onset,
                        "delay_sec": delay,
                    })
            if not hit:
                emission_results.append({
                    "text": em["text"],
                    "t_start": t_s, "t_end": t_e,
                    "hit": False, "missed": True,
                    "match_type": None, "exact_match": False,
                    "pred_text": None, "pred_onset_sec": None,
                    "delay_sec": None,
                })

        delays = [
            er["delay_sec"] for er in emission_results
            if er.get("hit") and er.get("delay_sec") is not None
        ]
        n_false_alarm = len(pred_runs) - len(matched_preds)
        return {
            "tp": tp,
            "n_false_alarm": n_false_alarm,
            "emission_results": emission_results,
            "run_fate": run_fate,
            "run_meta": run_meta,
            "delays": delays,
        }

    def compute_batch(
        self,
        predictions: List[TimelineItem],
        references: List[TimelineItem],
        output_dir: Union[str, Path] = None,
        tokenizer=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = []
        for pred, ref in zip(predictions, references):
            sample_id = ref.sample_id
            silence_id = ref.silence_token_id
            prefix_len = ref.prefix_len
            timeline_len = ref.timeline_len
            chunks_per_second = ref.chunks_per_second

            ref_seq = ref.ref_full_text_tokens
            pred_seq = pred.pred_full_text_tokens
            assert pred_seq is not None
            assert chunks_per_second is not None and chunks_per_second > 0
            assert silence_id is not None
            assert prefix_len is not None

            L = int(timeline_len) if timeline_len is not None else int(ref_seq.shape[0])
            # Truncate ref_seq to L for backward compat (when ref was provided)
            if ref_seq is not None:
                ref_seq = ref_seq[:L]
            else:
                ref_seq = torch.full((L,), silence_id, dtype=torch.long)
            # Predictions are never truncated: content beyond the declared
            # timeline_len (late answers, trailing false alarms) is scored.
            if pred_seq.shape[0] < L:
                # Pad pred with wait if shorter than the timeline (missed tail).
                pad = torch.full((L - pred_seq.shape[0],), silence_id, dtype=pred_seq.dtype)
                pred_seq = torch.cat([pred_seq, pad], dim=0)

            ref_emissions = ref.ref_emissions or []
            units, _source = resolve_pred_units(
                pred, pred_seq=pred_seq, silence_id=silence_id, prefix_len=prefix_len,
                chunks_per_second=chunks_per_second, tokenizer=tokenizer,
                onset_source=self.onset_source, sample_id=sample_id,
            )
            if _source == "events":
                # Event units with no alphanumeric content (e.g. EOS-only
                # bursts) are not predictions; grid runs keep legacy
                # behavior where they remain FA-eligible.
                units = [u for u in units if self._alnum_compact(u["compact"])]
            pred_onsets_sec = [u["onset_sec"] for u in units]
            pred_texts = [u["text"] for u in units]
            pred_compacts = [u["compact"] for u in units]
            ref_compacts = [self._norm_compact(em["text"]) for em in ref_emissions]
            ref_t_start = [em.get("t_start") for em in ref_emissions]
            ref_t_end = [em.get("t_end") for em in ref_emissions]

            matchers = {"fuzzy": self._content_matches, "exact": self._exact_matches}
            windows = self.windows_for(chunks_per_second)
            per_grace: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for gname, gsec in windows:
                per_grace[gname] = {}
                for kind in self.MATCH_KINDS:
                    per_grace[gname][kind] = self._match_at_grace(
                        gsec, matchers[kind], ref_emissions, units,
                        pred_onsets_sec, pred_texts, pred_compacts,
                        ref_compacts, ref_t_start, ref_t_end, kind=kind,
                    )

            broadest_name = "plus2c"
            broad_fuzzy = per_grace[broadest_name]["fuzzy"]
            broad_exact = per_grace[broadest_name]["exact"]

            n_ref = len(ref_emissions)
            n_pred_runs = len(units)

            accuracy_fields: Dict[str, float] = {}
            for gname, _ in windows:
                for kind in self.MATCH_KINDS:
                    res = per_grace[gname][kind]
                    tp, fa = res["tp"], res["n_false_alarm"]
                    n_pred_units = tp + fa
                    if n_ref > 0:
                        recall = tp / n_ref
                        precision = (tp / n_pred_units) if n_pred_units else 0.0
                        all_hit = tp == n_ref
                        clean = all_hit and fa == 0
                    else:
                        recall = 1.0
                        precision = 1.0 if fa == 0 else 0.0
                        all_hit = fa == 0
                        clean = fa == 0
                    f1 = (
                        (2 * precision * recall / (precision + recall))
                        if (precision + recall) > 0 else 0.0
                    )
                    accuracy_fields[f"sample_accuracy_{kind}_{gname}"] = 1.0 if all_hit else 0.0
                    accuracy_fields[f"sample_clean_{kind}_{gname}"] = 1.0 if clean else 0.0
                    accuracy_fields[f"emission_recall_{kind}_{gname}"] = recall
                    accuracy_fields[f"emission_precision_{kind}_{gname}"] = precision
                    accuracy_fields[f"emission_f1_{kind}_{gname}"] = f1
                    accuracy_fields[f"n_tp_{kind}_{gname}"] = tp
                    accuracy_fields[f"n_false_alarm_{kind}_{gname}"] = fa
                    accuracy_fields[f"n_pred_units_{kind}_{gname}"] = n_pred_units
                    accuracy_fields[f"n_missed_{kind}_{gname}"] = n_ref - tp
                accuracy_fields[f"n_false_alarm_{gname}"] = accuracy_fields[f"n_false_alarm_fuzzy_{gname}"]
                accuracy_fields[f"n_pred_units_{gname}"] = accuracy_fields[f"n_pred_units_fuzzy_{gname}"]
                accuracy_fields[f"n_missed_{gname}"] = accuracy_fields[f"n_missed_fuzzy_{gname}"]

            n_content_match = broad_fuzzy["tp"]
            n_exact_match = broad_exact["tp"]
            n_fuzzy_only = sum(
                1 for i in range(n_ref)
                if broad_fuzzy["emission_results"][i]["hit"]
                and not broad_exact["emission_results"][i]["hit"]
            )
            n_false_alarm = broad_fuzzy["n_false_alarm"]
            n_missed = n_ref - n_content_match

            emissions_out = []
            for i, em in enumerate(ref_emissions):
                entry = {
                    "text": em["text"],
                    "t_start": ref_t_start[i],
                    "t_end": ref_t_end[i],
                    "levels": {},
                }
                for gname, _ in windows:
                    entry["levels"][gname] = {}
                    for kind in self.MATCH_KINDS:
                        er = per_grace[gname][kind]["emission_results"][i]
                        entry["levels"][gname][kind] = {
                            "hit": er["hit"],
                            "missed": er["missed"],
                            "match_type": er["match_type"],
                            "pred_text": er["pred_text"],
                            "pred_onset_sec": er["pred_onset_sec"],
                            "delay_sec": er["delay_sec"],
                        }
                emissions_out.append(entry)

            pred_runs_out = []
            for k in range(n_pred_runs):
                entry = {
                    "onset_sec": pred_onsets_sec[k],
                    "text": pred_texts[k],
                    "levels": {},
                }
                for gname, _ in windows:
                    entry["levels"][gname] = {}
                    for kind in self.MATCH_KINDS:
                        entry["levels"][gname][kind] = {
                            "fate": per_grace[gname][kind]["run_fate"][k],
                            "match_type": per_grace[gname][kind]["run_meta"][k],
                        }
                pred_runs_out.append(entry)

            # Per-grace delay summaries (hits only).
            delay_fields: Dict[str, Any] = {}
            for gname, _ in windows:
                for kind in self.MATCH_KINDS:
                    delays = per_grace[gname][kind]["delays"]
                    delay_fields[f"delays_{kind}_{gname}"] = delays
                    delay_fields[f"delay_mean_{kind}_{gname}"] = (
                        float(np.mean(delays)) if delays else float("nan")
                    )
            # Headline delay: broadest fuzzy (same window family as FDR).
            broad_delays = broad_fuzzy["delays"]
            delay_fields["delays"] = broad_delays
            delay_fields["delay_mean"] = (
                float(np.mean(broad_delays)) if broad_delays else float("nan")
            )

            # FA density over the scored extent, not just the declared horizon.
            scored_frames = max(L, int(pred_seq.shape[0]))
            content_groups = max(0, scored_frames - prefix_len)
            _fa_tp = n_false_alarm + n_content_match
            fdr = (n_false_alarm / _fa_tp) if _fa_tp > 0 else 0.0
            sample_result = {
                "sample_id": sample_id,
                "prompt": ref.prompt,
                "timeline_len": L,
                "prefix_len": prefix_len,
                "chunks_per_second": chunks_per_second,
                "onset_source": _source,
                "grace_windows": [
                    [gname, gsec] for gname, gsec in windows
                ],
                "n_ref_emissions": n_ref,
                "is_negative": n_ref == 0,
                "n_content_match": n_content_match,
                "n_exact_match": n_exact_match,
                "n_fuzzy_match": n_fuzzy_only,
                "n_missed": n_missed,
                "total_pred_runs": n_pred_runs,
                "total_pred_units": n_content_match + n_false_alarm,
                "n_false_alarm": n_false_alarm,
                "n_false_alarm_exact": broad_exact["n_false_alarm"],
                "fdr": fdr,
                "false_alarm_density": (n_false_alarm / content_groups) if content_groups > 0 else 0.0,
                "exact_match_rate": (n_exact_match / n_content_match) if n_content_match > 0 else 0.0,
                **accuracy_fields,
                **delay_fields,
                "pred_full_text": pred.pred_full_text,
                "ref_full_text": ref.ref_full_text or "",
                "emissions": emissions_out,
                "pred_runs": pred_runs_out,
            }

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                dump_path = os.path.abspath(os.path.join(output_dir, f"{sample_id}_streaming.json"))
                with open(dump_path, "w") as f:
                    json.dump(json_safe(sample_result), f, indent=2)
                sample_result["dump_path"] = dump_path

            results.append(sample_result)
        return results

    def compute_dataset_level_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not sample_results:
            return {}
        n = len(sample_results)
        gnames = list(self.WINDOW_NAMES)
        mean_keys = (
            [f"sample_accuracy_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"sample_clean_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"n_tp_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"n_false_alarm_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"n_pred_units_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"n_missed_{kind}_{name}" for kind in self.MATCH_KINDS for name in gnames]
            + [f"n_false_alarm_{name}" for name in gnames]
            + [f"n_pred_units_{name}" for name in gnames]
            + [f"n_missed_{name}" for name in gnames]
            + ["false_alarm_density"]
            # exact_match_rate / fdr are micro-averaged below
        )
        agg: Dict[str, Any] = {}
        for k in mean_keys:
            vals = [
                r.get(k) for r in sample_results
                if r.get(k) is not None and not (
                    isinstance(r.get(k), float) and math.isnan(r.get(k))
                )
            ]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg["n_samples"] = n
        agg["total_ref_emissions"] = int(sum(r.get("n_ref_emissions", 0) for r in sample_results))
        agg["total_content_match"] = int(sum(r.get("n_content_match", 0) for r in sample_results))
        agg["total_exact_match"] = int(sum(r.get("n_exact_match", 0) for r in sample_results))
        agg["total_missed"] = int(sum(r.get("n_missed", 0) for r in sample_results))
        agg["total_false_alarm"] = int(sum(r.get("n_false_alarm", 0) for r in sample_results))
        _content = agg["total_content_match"]
        agg["exact_match_rate"] = (
            (agg["total_exact_match"] / _content) if _content > 0 else 0.0
        )
        _fa_tp = agg["total_false_alarm"] + agg["total_content_match"]
        agg["fdr"] = (agg["total_false_alarm"] / _fa_tp) if _fa_tp > 0 else 0.0
        n_ref_total = agg["total_ref_emissions"]
        for name in gnames:
            for kind in self.MATCH_KINDS:
                tp = int(sum(r.get(f"n_tp_{kind}_{name}", 0) for r in sample_results))
                fa = int(sum(r.get(f"n_false_alarm_{kind}_{name}", 0) for r in sample_results))
                agg[f"total_tp_{kind}_{name}"] = tp
                agg[f"total_false_alarm_{kind}_{name}"] = fa
                n_pred_units = tp + fa
                if n_ref_total > 0:
                    recall = tp / n_ref_total
                    precision = (tp / n_pred_units) if n_pred_units else 0.0
                else:
                    recall = 1.0
                    precision = 1.0 if fa == 0 else 0.0
                f1 = (
                    (2 * precision * recall / (precision + recall))
                    if (precision + recall) > 0 else 0.0
                )
                agg[f"emission_recall_{kind}_{name}"] = recall
                agg[f"emission_precision_{kind}_{name}"] = precision
                agg[f"emission_f1_{kind}_{name}"] = f1
            agg[f"total_false_alarm_{name}"] = agg[f"total_false_alarm_fuzzy_{name}"]

        # Delay aggregates over hits (per grace × kind + headline broadest fuzzy).
        for name in gnames:
            for kind in self.MATCH_KINDS:
                delays: List[float] = []
                for r in sample_results:
                    delays.extend(r.get(f"delays_{kind}_{name}") or [])
                agg[f"n_delay_{kind}_{name}"] = len(delays)
                agg[f"delay_mean_{kind}_{name}"] = (
                    float(np.mean(delays)) if delays else float("nan")
                )
        broadest_name = "plus2c"
        headline_delays: List[float] = []
        for r in sample_results:
            headline_delays.extend(r.get(f"delays_fuzzy_{broadest_name}") or [])
        agg["n_delay"] = len(headline_delays)
        agg["delay_mean"] = (
            float(np.mean(headline_delays)) if headline_delays else float("nan")
        )
        return agg
