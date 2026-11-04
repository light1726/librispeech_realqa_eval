"""Post-audio / open-latency evaluation metrics.

Separate from ``StreamingMetrics`` (streaming grace-window matching).

Targets and predictions are aligned with an **ordered exclusive** matcher:
each predicted non-wait run is one unit and is assigned to at most one
reference emission; order is preserved; a prediction may not start much
before its target.

Latency for each hit::

    delay_sec = pred_onset_sec - ref_t_start

A multi-word prediction run is kept intact (not split into tokens), so one
blob can cover at most one target under exclusive assignment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from realqa.metrics import (
    ONSET_SOURCES,
    StreamingMetrics,
    TimelineItem,
    json_safe,
    resolve_pred_units,
)


@dataclass
class PredUnit:
    """One predicted content unit = one non-wait run on the timeline."""

    text: str
    compact: str
    onset_sec: float
    run_index: int


class PostAudioMetrics:
    """Ordered exclusive alignment + absolute onset delay for post-audio APIs."""

    MATCH_KINDS: Tuple[str, ...] = ("fuzzy", "exact")

    def __init__(self, early_eps_sec: float = 0.05, onset_source: str = "auto"):
        """
        Parameters
        ----------
        early_eps_sec :
            Predictions with ``onset < ref_t_start - early_eps_sec`` are
            rejected for that target (too early).
        onset_source :
            ``auto`` (events per row when ``pred_runs`` present, else grid),
            ``events`` (require ``pred_runs``), or ``grid`` (always derive
            units from the token grid).
        """
        self.early_eps_sec = float(early_eps_sec)
        if onset_source not in ONSET_SOURCES:
            raise ValueError(
                f"onset_source must be one of {ONSET_SOURCES}, got {onset_source!r}"
            )
        self.onset_source = onset_source

    # ── public API ──────────────────────────────────────────────

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
            sample = self._compute_sample(pred, ref, tokenizer=tokenizer)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                dump_path = os.path.abspath(
                    os.path.join(output_dir, f"{ref.sample_id}_post_audio.json")
                )
                with open(dump_path, "w") as f:
                    json.dump(json_safe(sample), f, indent=2)
                sample["dump_path"] = dump_path
            results.append(sample)
        return results

    def compute_dataset_level_metrics(
        self, sample_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not sample_results:
            return {}
        n = len(sample_results)
        agg: Dict[str, Any] = {"n_samples": n}

        for kind in self.MATCH_KINDS:
            tp = int(sum(r.get(f"n_tp_{kind}", 0) for r in sample_results))
            fa = int(sum(r.get(f"n_false_alarm_{kind}", 0) for r in sample_results))
            missed = int(sum(r.get(f"n_missed_{kind}", 0) for r in sample_results))
            n_ref = int(sum(r.get("n_ref_emissions", 0) for r in sample_results))
            n_pred_units = tp + fa
            if n_ref > 0:
                recall = tp / n_ref
                precision = (tp / n_pred_units) if n_pred_units else 0.0
            else:
                recall = 1.0
                precision = 1.0 if fa == 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            agg[f"total_tp_{kind}"] = tp
            agg[f"total_false_alarm_{kind}"] = fa
            agg[f"total_missed_{kind}"] = missed
            agg[f"emission_recall_{kind}"] = recall
            agg[f"emission_precision_{kind}"] = precision
            agg[f"emission_f1_{kind}"] = f1
            _fa_tp = fa + tp
            agg[f"fdr_{kind}"] = (fa / _fa_tp) if _fa_tp > 0 else 0.0

            delays: List[float] = []
            for r in sample_results:
                delays.extend(r.get(f"delays_{kind}") or [])
            agg[f"n_delay_{kind}"] = len(delays)
            agg[f"delay_mean_{kind}"] = (
                float(np.mean(delays)) if delays else float("nan")
            )

        # Headline aliases (fuzzy)
        agg["total_ref_emissions"] = int(
            sum(r.get("n_ref_emissions", 0) for r in sample_results)
        )
        agg["total_content_match"] = agg["total_tp_fuzzy"]
        agg["total_exact_match"] = agg["total_tp_exact"]
        agg["total_missed"] = agg["total_missed_fuzzy"]
        agg["total_false_alarm"] = agg["total_false_alarm_fuzzy"]
        agg["fdr"] = agg["fdr_fuzzy"]
        agg["emission_f1_fuzzy"] = agg["emission_f1_fuzzy"]
        agg["delay_mean"] = agg["delay_mean_fuzzy"]
        return agg

    # ── per-sample ──────────────────────────────────────────────

    def _compute_sample(
        self,
        pred: TimelineItem,
        ref: TimelineItem,
        tokenizer=None,
    ) -> Dict[str, Any]:
        sample_id = ref.sample_id
        silence_id = ref.silence_token_id
        prefix_len = ref.prefix_len
        timeline_len = ref.timeline_len
        chunks_per_second = ref.chunks_per_second

        assert pred.pred_full_text_tokens is not None
        assert chunks_per_second is not None and chunks_per_second > 0
        assert silence_id is not None
        assert prefix_len is not None

        L = int(timeline_len) if timeline_len is not None else int(
            pred.pred_full_text_tokens.shape[0]
        )
        # Predictions are never truncated: post-audio answers routinely land
        # beyond the declared timeline_len (open-latency protocol).
        pred_seq = pred.pred_full_text_tokens
        if pred_seq.shape[0] < L:
            pad = torch.full((L - pred_seq.shape[0],), silence_id, dtype=pred_seq.dtype)
            pred_seq = torch.cat([pred_seq, pad], dim=0)

        ref_emissions = list(ref.ref_emissions or [])
        # Stable order by t_start (then original index).
        order = sorted(
            range(len(ref_emissions)),
            key=lambda i: (
                float("inf")
                if ref_emissions[i].get("t_start") is None
                else float(ref_emissions[i]["t_start"]),
                i,
            ),
        )
        ref_emissions = [ref_emissions[i] for i in order]

        raw_units, _source = resolve_pred_units(
            pred, pred_seq=pred_seq, silence_id=silence_id, prefix_len=prefix_len,
            chunks_per_second=chunks_per_second, tokenizer=tokenizer,
            onset_source=self.onset_source, sample_id=sample_id,
        )
        pred_units = [
            PredUnit(
                text=u["text"],
                compact=u["compact"],
                onset_sec=u["onset_sec"],
                run_index=i,
            )
            for i, u in enumerate(raw_units)
            if StreamingMetrics._alnum_compact(u["compact"])
        ]

        per_kind: Dict[str, Dict[str, Any]] = {}
        for kind in self.MATCH_KINDS:
            per_kind[kind] = self._align(
                ref_emissions, pred_units, kind=kind,
            )

        fuzzy = per_kind["fuzzy"]
        exact = per_kind["exact"]
        n_ref = len(ref_emissions)
        n_pred = len(pred_units)

        emissions_out = []
        for i, em in enumerate(ref_emissions):
            entry = {
                "text": em["text"],
                "t_start": em.get("t_start"),
                "t_end": em.get("t_end"),
                "levels": {},
            }
            for kind in self.MATCH_KINDS:
                er = per_kind[kind]["emission_results"][i]
                entry["levels"][kind] = {
                    "hit": er["hit"],
                    "missed": er["missed"],
                    "match_type": er["match_type"],
                    "pred_text": er["pred_text"],
                    "pred_onset_sec": er["pred_onset_sec"],
                    "delay_sec": er["delay_sec"],
                }
            emissions_out.append(entry)

        pred_units_out = []
        for k, u in enumerate(pred_units):
            entry = {
                "onset_sec": u.onset_sec,
                "text": u.text,
                "run_index": u.run_index,
                "levels": {},
            }
            for kind in self.MATCH_KINDS:
                entry["levels"][kind] = {
                    "fate": per_kind[kind]["unit_fate"][k],
                    "match_type": per_kind[kind]["unit_meta"][k],
                }
            pred_units_out.append(entry)

        sample: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt": ref.prompt,
            "metric_mode": "post_audio",
            "timeline_len": L,
            "prefix_len": prefix_len,
            "chunks_per_second": chunks_per_second,
            "onset_source": _source,
            "early_eps_sec": self.early_eps_sec,
            "n_ref_emissions": n_ref,
            "is_negative": n_ref == 0,
            "total_pred_units": n_pred,
            "n_content_match": fuzzy["tp"],
            "n_exact_match": exact["tp"],
            "n_missed": fuzzy["n_missed"],
            "n_false_alarm": fuzzy["n_false_alarm"],
            "pred_full_text": pred.pred_full_text,
            "ref_full_text": ref.ref_full_text or "",
            "emissions": emissions_out,
            "pred_units": pred_units_out,
        }
        for kind in self.MATCH_KINDS:
            res = per_kind[kind]
            tp, fa, missed = res["tp"], res["n_false_alarm"], res["n_missed"]
            n_pred_units = tp + fa
            if n_ref > 0:
                recall = tp / n_ref
                precision = (tp / n_pred_units) if n_pred_units else 0.0
            else:
                recall = 1.0
                precision = 1.0 if fa == 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            sample[f"n_tp_{kind}"] = tp
            sample[f"n_false_alarm_{kind}"] = fa
            sample[f"n_missed_{kind}"] = missed
            sample[f"emission_recall_{kind}"] = recall
            sample[f"emission_precision_{kind}"] = precision
            sample[f"emission_f1_{kind}"] = f1
            sample[f"delays_{kind}"] = res["delays"]
            sample[f"delay_mean_{kind}"] = (
                float(np.mean(res["delays"])) if res["delays"] else float("nan")
            )
            _fa_tp = fa + tp
            sample[f"fdr_{kind}"] = (fa / _fa_tp) if _fa_tp > 0 else 0.0
        sample["fdr"] = sample["fdr_fuzzy"]
        sample["delay_mean"] = sample["delay_mean_fuzzy"]
        return sample

    # ── alignment ───────────────────────────────────────────────

    def _align(
        self,
        ref_emissions: List[Dict[str, Any]],
        pred_units: List[PredUnit],
        *,
        kind: str,
    ) -> Dict[str, Any]:
        match_fn = (
            StreamingMetrics._exact_matches
            if kind == "exact"
            else StreamingMetrics._content_matches
        )
        n_pred = len(pred_units)
        unit_fate = ["fa"] * n_pred
        unit_meta: List[Optional[str]] = [None] * n_pred
        emission_results = []
        delays: List[float] = []
        tp = 0
        cursor = 0  # next eligible pred unit (monotonic)

        for em in ref_emissions:
            rc = StreamingMetrics._norm_compact(em.get("text") or "")
            t_s = em.get("t_start")
            t_e = em.get("t_end")
            lo = (float(t_s) - self.early_eps_sec) if t_s is not None else float("-inf")

            hit_k: Optional[int] = None
            hit_exact = False
            for k in range(cursor, n_pred):
                onset = pred_units[k].onset_sec
                # Too early for this (and any later) target → leave as FA.
                if onset < lo - 1e-9:
                    continue
                matched, is_exact = match_fn(rc, pred_units[k].compact)
                if not matched:
                    continue
                hit_k = k
                hit_exact = is_exact
                break

            if hit_k is not None and t_s is not None:
                # Label dump match_type from content_matches on the unit.
                _, is_exact_label = StreamingMetrics._content_matches(
                    rc, pred_units[hit_k].compact
                )
                match_type = "exact" if is_exact_label else "fuzzy"
                delay = pred_units[hit_k].onset_sec - float(t_s)
                unit_fate[hit_k] = "tp"
                unit_meta[hit_k] = match_type
                cursor = hit_k + 1
                tp += 1
                delays.append(delay)
                emission_results.append({
                    "text": em["text"],
                    "t_start": t_s,
                    "t_end": t_e,
                    "hit": True,
                    "missed": False,
                    "match_type": match_type,
                    "exact_match": hit_exact if kind == "exact" else is_exact_label,
                    "pred_text": pred_units[hit_k].text,
                    "pred_onset_sec": pred_units[hit_k].onset_sec,
                    "delay_sec": delay,
                })
            else:
                emission_results.append({
                    "text": em["text"],
                    "t_start": t_s,
                    "t_end": t_e,
                    "hit": False,
                    "missed": True,
                    "match_type": None,
                    "exact_match": False,
                    "pred_text": None,
                    "pred_onset_sec": None,
                    "delay_sec": None,
                })

        n_false_alarm = sum(1 for f in unit_fate if f == "fa")
        n_missed = len(ref_emissions) - tp
        return {
            "tp": tp,
            "n_false_alarm": n_false_alarm,
            "n_missed": n_missed,
            "emission_results": emission_results,
            "unit_fate": unit_fate,
            "unit_meta": unit_meta,
            "delays": delays,
        }
