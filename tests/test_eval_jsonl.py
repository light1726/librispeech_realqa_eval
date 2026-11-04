import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realqa.eval import build_items, main
from realqa.metrics import StreamingMetrics
from realqa.post_audio_metrics import PostAudioMetrics


def test_demo_jsonl_perfect_and_fa(tmp_path):
    demo = ROOT / "examples" / "demo.jsonl"
    out = tmp_path / "out"
    rc = main([
        str(demo),
        "--output-dir", str(out),
        "--silence-token", "<silence>",
        "--eos-token", "<|im_end|>",
        "--chunks-per-second", "6.25",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    m = results["dataset_level_metrics"]["streaming_metrics"]
    assert m["n_samples"] == 3
    # 2 positive refs (demo-perfect + demo-false-alarm), both hit under fuzzy
    assert m["total_ref_emissions"] == 2
    assert m["total_content_match"] == 2
    # one FA from demo-false-alarm
    assert m["total_false_alarm"] == 1
    assert m["emission_f1_fuzzy_during_word"] > 0.7


def test_build_items_token_lists():
    rows = [{
        "sample_id": "s0",
        "prefix_len": 1,
        "pred": ["<w>", "hi", "<e>", "<w>"],
        "ref": ["<w>", "hi", "<e>", "<w>"],
        "emissions": [{"text": "hi", "start_group": 0, "end_group": 0}],
    }]
    preds, refs, tok = build_items(
        rows,
        silence_token="<w>",
        eos_tokens=["<e>"],
        default_chunks_per_second=6.25,
    )
    assert tok.silence_token_id == 0
    assert refs[0].prefix_len == 1
    assert abs(refs[0].ref_emissions[0]["t_end"] - 0.0) < 1e-9
    metric = StreamingMetrics()
    sample = metric.compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["emission_f1_fuzzy_during_word"] == 1.0
    assert sample["n_false_alarm"] == 0
    assert "task_subtype" not in sample
    assert sample["fdr"] == 0.0
    # onset at group 0 → delay = 0 - 0 = 0
    assert sample["emissions"][0]["levels"]["during_word"]["fuzzy"]["delay_sec"] == 0.0
    assert sample["delay_mean"] == 0.0
    ds = metric.compute_dataset_level_metrics([sample])
    assert ds["delay_mean"] == 0.0
    assert ds["n_delay"] == 1


def test_over_emission_merge_is_fuzzy():
    """Two in-window exact 'hi' runs merge to 'hihi' → fuzzy match_type."""
    # Timeline: prefix wait, then hi, wait, hi within the same word window.
    # Groups at positions 1 and 3 → onsets (1-1)/6.25=0 and (3-1)/6.25=0.32
    # Emission window [0, 0.48] covers both.
    rows = [{
        "sample_id": "overemit",
        "prefix_len": 1,
        "chunks_per_second": 6.25,
        "pred": ["<w>", "hi", "<w>", "hi", "<w>"],
        "ref": ["<w>", "hi", "<w>", "<w>", "<w>"],
        "emissions": [{"text": "hi", "t_start": 0.0, "t_end": 0.48}],
    }]
    preds, refs, tok = build_items(
        rows,
        silence_token="<w>",
        eos_tokens=["<e>"],
        default_chunks_per_second=6.25,
    )
    sample = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["n_content_match"] == 1
    assert sample["n_false_alarm"] == 0  # both runs absorbed into one TP
    em = sample["emissions"][0]["levels"]["plus2c"]["fuzzy"]
    assert em["hit"] is True
    assert em["match_type"] == "fuzzy"
    assert em["pred_text"] == "hi hi"
    assert abs(em["delay_sec"] - 0.0) < 1e-9  # combined onset = min(0, 0.32) = 0
    # Exact pass: the merged unit ("hihi") is not form-equal to "hi", so the
    # exact tier does NOT credit the emission — summary and detail agree that
    # over-emission is fuzzy, not exact.
    em_ex = sample["emissions"][0]["levels"]["plus2c"]["exact"]
    assert em_ex["hit"] is False
    assert em_ex["missed"] is True
    assert sample["n_exact_match"] == 0
    assert sample["exact_match_rate"] == 0.0
    assert sample["n_fuzzy_match"] == 1


def test_exact_match_rate_dataset_is_micro():
    """Dataset exact_match_rate is pooled, not mean of per-sample rates."""
    metric = StreamingMetrics()
    pos_rows = [{
        "sample_id": "pos",
        "prefix_len": 1,
        "pred": ["<w>", "hi", "<w>"],
        "ref": ["<w>", "hi", "<w>"],
        "emissions": [{"text": "hi", "t_start": 0.0, "t_end": 0.16}],
    }]
    neg_rows = [{
        "sample_id": "neg",
        "prefix_len": 1,
        "pred": ["<w>", "<w>", "<w>"],
        "ref": ["<w>", "<w>", "<w>"],
        "emissions": [],
    }]
    pos_p, pos_r, tok = build_items(
        pos_rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=6.25
    )
    neg_p, neg_r, _ = build_items(
        neg_rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=6.25
    )
    pos = metric.compute_batch(pos_p, pos_r, tokenizer=tok)[0]
    neg = metric.compute_batch(neg_p, neg_r, tokenizer=tok)[0]
    assert abs(pos["exact_match_rate"] - 1.0) < 1e-9
    assert neg["exact_match_rate"] == 0.0
    agg = metric.compute_dataset_level_metrics([pos, neg])
    assert abs(agg["exact_match_rate"] - 1.0) < 1e-9
    assert abs(agg["fdr"] - 0.0) < 1e-9


def test_number_word_digit_equivalence():
    m = StreamingMetrics
    assert m._content_matches("one", "1") == (True, True)
    assert m._content_matches("1", "one") == (True, True)
    assert m._content_matches("two", "2") == (True, True)
    assert m._exact_matches("three", "3") == (True, True)
    # still fuzzy for containment
    assert m._content_matches("one", "number1")[0] is True
    assert m._content_matches("one", "number1")[1] is False
    # unrelated still fails
    assert m._content_matches("one", "two") == (False, False)


# ═══════════════════════════════════════════════════════════════════
# pred_runs (event-level onsets)
# ═══════════════════════════════════════════════════════════════════


def _grid_row(pred, emissions, **extra):
    row = {
        "sample_id": "s",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "pred": pred,
        "emissions": emissions,
    }
    row.update(extra)
    return row


def test_pred_runs_exact_onset_survives_quantization():
    """Answer at t_start=1.44 floors to frame onset 1.4 (< t_start) on the
    grid; the event-level onset keeps it an in-window hit with delay 0."""
    rows = [{
        "sample_id": "ontime",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "pred": ["<w>"] * 7 + ["stew"] + ["<w>"] * 3,
        "pred_runs": [{"text": "stew", "t_start": 1.44, "t_end": 1.6}],
        "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}],
    }]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    # Grid fallback (old behavior): onset 1.4 < t_start → miss + FA.
    grid = StreamingMetrics(onset_source="grid").compute_batch(
        preds, refs, tokenizer=tok
    )[0]
    assert grid["n_content_match"] == 0
    assert grid["n_false_alarm"] == 1
    # Events (auto because pred_runs present): exact onset → hit, delay 0.
    ev = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert ev["onset_source"] == "events"
    assert ev["n_content_match"] == 1
    assert ev["n_false_alarm"] == 0
    em = ev["emissions"][0]["levels"]["during_word"]["fuzzy"]
    assert em["hit"] is True
    assert abs(em["delay_sec"]) < 1e-9


def test_pred_runs_burst_identity_not_frame_contiguity():
    """One burst whose deltas are 0.6 s apart stays one unit; two bursts in
    adjacent frames stay two units (grid contiguity would reverse both)."""
    rows = [{
        "sample_id": "bursts",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        # grid shows "wes" @0.2, "ley" @0.8 (split) — events say one "wesley"
        "pred": ["<w>", "wes", "<w>", "<w>", "ley", "<w>"] * 1 + ["<w>"] * 2,
        "pred_runs": [
            {"text": "wesley", "t_start": 0.2, "t_end": 0.9},
            {"text": "one", "t_start": 1.2, "t_end": 1.25},
            {"text": "two", "t_start": 1.4, "t_end": 1.45},
        ],
        "emissions": [
            {"text": "wesley", "t_start": 0.2, "t_end": 0.8},
            {"text": "one", "t_start": 1.15, "t_end": 1.25},
            {"text": "two", "t_start": 1.35, "t_end": 1.45},
        ],
    }]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    ev = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert ev["onset_source"] == "events"
    assert ev["n_content_match"] == 3
    assert ev["n_false_alarm"] == 0
    assert [r["text"] for r in ev["pred_runs"]] == ["wesley", "one", "two"]

    # post_audio: two adjacent-frame bursts remain two exclusive units.
    pa = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert pa["onset_source"] == "events"
    assert pa["n_tp_fuzzy"] == 3
    assert pa["n_false_alarm_fuzzy"] == 0


def test_onset_source_events_requires_pred_runs():
    rows = [_grid_row(["<w>", "hi", "<w>"],
                      [{"text": "hi", "t_start": 0.2, "t_end": 0.4}])]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    with pytest.raises(ValueError, match="pred_runs"):
        StreamingMetrics(onset_source="events").compute_batch(preds, refs, tokenizer=tok)


def test_pred_runs_invalid_entry_raises():
    rows = [{
        "sample_id": "bad", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>"], "pred_runs": [{"text": "hi"}],
        "emissions": [{"text": "hi", "t_start": 0.0, "t_end": 0.1}],
    }]
    with pytest.raises(ValueError, match="pred_runs"):
        build_items(rows, silence_token="<w>", eos_tokens=["<e>"],
                    default_chunks_per_second=5.0)


def test_cli_onset_source_flag(tmp_path):
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({
        "sample_id": "cli", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>"] * 7 + ["stew"] + ["<w>"] * 3,
        "pred_runs": [{"text": "stew", "t_start": 1.44, "t_end": 1.6}],
        "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>",
        "--chunks-per-second", "5.0",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    assert results["config"]["onset_source"] == "auto"
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    assert sample["onset_source"] == "events"
    assert sample["n_content_match"] == 1

    # grid override restores the legacy quantized behavior
    out2 = tmp_path / "out-grid"
    rc2 = main([
        str(src), "--output-dir", str(out2),
        "--silence-token", "<w>", "--eos-token", "<e>",
        "--chunks-per-second", "5.0", "--onset-source", "grid",
    ])
    assert rc2 == 0
    results2 = json.loads((out2 / "all_results.json").read_text())
    sample2 = results2["sample_level_metrics"]["streaming_metrics"][0]
    assert sample2["onset_source"] == "grid"
    assert sample2["n_content_match"] == 0


# ═══════════════════════════════════════════════════════════════════
# acceptance windows: always during_word(0) / +1c(C) / +2c(2C),
# C = 1 / chunks_per_second per row
# ═══════════════════════════════════════════════════════════════════


def test_windows_derived_from_chunks_per_second(tmp_path):
    """C comes from the row's chunks_per_second (2.5 -> C=0.4); a late
    answer inside +2c but beyond +1c hits only under plus2c."""
    src = tmp_path / "paper.jsonl"
    # ref word [0.2, 0.4]; answer onset 1.0 (= t_end + 0.6):
    # during_word [0.2, 0.4] miss; +1c [0.2, 0.8] miss; +2c [0.2, 1.2] hit
    src.write_text(json.dumps({
        "sample_id": "p1", "prefix_len": 0, "chunks_per_second": 2.5,
        "pred": ["<w>"] * 5 + ["dog"] + ["<w>"] * 3,
        "pred_runs": [{"text": "dog", "t_start": 1.0, "t_end": 1.1}],
        "emissions": [{"text": "dog", "t_start": 0.2, "t_end": 0.4}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    assert sample["grace_windows"] == [
        ["during_word", 0.0], ["plus1c", 0.4], ["plus2c", 0.8],
    ]
    levels = sample["emissions"][0]["levels"]
    assert levels["during_word"]["fuzzy"]["hit"] is False
    assert levels["plus1c"]["fuzzy"]["hit"] is False
    assert levels["plus2c"]["fuzzy"]["hit"] is True


def test_window_protocol_names(tmp_path):
    """Every sample reports exactly the three protocol windows with g
    derived from its chunk rate (6.25 -> 0 / 0.16 / 0.32)."""
    src = tmp_path / "proto.jsonl"
    src.write_text(json.dumps({
        "sample_id": "f1", "prefix_len": 0, "chunks_per_second": 6.25,
        "pred": ["<w>", "hi", "<w>"],
        "emissions": [{"text": "hi", "t_start": 0.2, "t_end": 0.3}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    assert sample["grace_windows"] == [
        ["during_word", 0.0], ["plus1c", 0.16], ["plus2c", 0.32],
    ]
    assert set(sample["emissions"][0]["levels"].keys()) == {
        "during_word", "plus1c", "plus2c",
    }


# ═══════════════════════════════════════════════════════════════════
# pred_runs (event-level onsets)
# ═══════════════════════════════════════════════════════════════════


def _grid_row(pred, emissions, **extra):
    row = {
        "sample_id": "s",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "pred": pred,
        "emissions": emissions,
    }
    row.update(extra)
    return row


def test_pred_runs_exact_onset_survives_quantization():
    """Answer at t_start=1.44 floors to frame onset 1.4 (< t_start) on the
    grid; the event-level onset keeps it an in-window hit with delay 0."""
    rows = [{
        "sample_id": "ontime",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "pred": ["<w>"] * 7 + ["stew"] + ["<w>"] * 3,
        "pred_runs": [{"text": "stew", "t_start": 1.44, "t_end": 1.6}],
        "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}],
    }]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    # Grid fallback (old behavior): onset 1.4 < t_start → miss + FA.
    grid = StreamingMetrics(onset_source="grid").compute_batch(
        preds, refs, tokenizer=tok
    )[0]
    assert grid["n_content_match"] == 0
    assert grid["n_false_alarm"] == 1
    # Events (auto because pred_runs present): exact onset → hit, delay 0.
    ev = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert ev["onset_source"] == "events"
    assert ev["n_content_match"] == 1
    assert ev["n_false_alarm"] == 0
    em = ev["emissions"][0]["levels"]["during_word"]["fuzzy"]
    assert em["hit"] is True
    assert abs(em["delay_sec"]) < 1e-9


def test_pred_runs_burst_identity_not_frame_contiguity():
    """One burst whose deltas are 0.6 s apart stays one unit; two bursts in
    adjacent frames stay two units (grid contiguity would reverse both)."""
    rows = [{
        "sample_id": "bursts",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        # grid shows "wes" @0.2, "ley" @0.8 (split) — events say one "wesley"
        "pred": ["<w>", "wes", "<w>", "<w>", "ley", "<w>"] * 1 + ["<w>"] * 2,
        "pred_runs": [
            {"text": "wesley", "t_start": 0.2, "t_end": 0.9},
            {"text": "one", "t_start": 1.2, "t_end": 1.25},
            {"text": "two", "t_start": 1.4, "t_end": 1.45},
        ],
        "emissions": [
            {"text": "wesley", "t_start": 0.2, "t_end": 0.8},
            {"text": "one", "t_start": 1.15, "t_end": 1.25},
            {"text": "two", "t_start": 1.35, "t_end": 1.45},
        ],
    }]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    ev = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert ev["onset_source"] == "events"
    assert ev["n_content_match"] == 3
    assert ev["n_false_alarm"] == 0
    assert [r["text"] for r in ev["pred_runs"]] == ["wesley", "one", "two"]

    # post_audio: two adjacent-frame bursts remain two exclusive units.
    pa = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert pa["onset_source"] == "events"
    assert pa["n_tp_fuzzy"] == 3
    assert pa["n_false_alarm_fuzzy"] == 0


def test_onset_source_events_requires_pred_runs():
    rows = [_grid_row(["<w>", "hi", "<w>"],
                      [{"text": "hi", "t_start": 0.2, "t_end": 0.4}])]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    with pytest.raises(ValueError, match="pred_runs"):
        StreamingMetrics(onset_source="events").compute_batch(preds, refs, tokenizer=tok)


def test_pred_runs_invalid_entry_raises():
    rows = [{
        "sample_id": "bad", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>"], "pred_runs": [{"text": "hi"}],
        "emissions": [{"text": "hi", "t_start": 0.0, "t_end": 0.1}],
    }]
    with pytest.raises(ValueError, match="pred_runs"):
        build_items(rows, silence_token="<w>", eos_tokens=["<e>"],
                    default_chunks_per_second=5.0)


def test_cli_onset_source_flag(tmp_path):
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({
        "sample_id": "cli", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>"] * 7 + ["stew"] + ["<w>"] * 3,
        "pred_runs": [{"text": "stew", "t_start": 1.44, "t_end": 1.6}],
        "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>",
        "--chunks-per-second", "5.0",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    assert results["config"]["onset_source"] == "auto"
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    assert sample["onset_source"] == "events"
    assert sample["n_content_match"] == 1

    # grid override restores the legacy quantized behavior
    out2 = tmp_path / "out-grid"
    rc2 = main([
        str(src), "--output-dir", str(out2),
        "--silence-token", "<w>", "--eos-token", "<e>",
        "--chunks-per-second", "5.0", "--onset-source", "grid",
    ])
    assert rc2 == 0
    results2 = json.loads((out2 / "all_results.json").read_text())
    sample2 = results2["sample_level_metrics"]["streaming_metrics"][0]
    assert sample2["onset_source"] == "grid"
    assert sample2["n_content_match"] == 0


# ═══════════════════════════════════════════════════════════════════
# grace scheme (+1c / +2c paper windows)
# ═══════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════
# timeline_len: predictions are never truncated
# ═══════════════════════════════════════════════════════════════════


def test_pred_beyond_timeline_len_is_scored():
    """A non-silence token past the declared timeline_len must not silently
    disappear — it is a false alarm and must be counted."""
    rows = [{
        "sample_id": "cut",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 2,
        "pred": ["hi", "<w>", "wrong"],
        "emissions": [{"text": "hi", "t_start": 0.0, "t_end": 0.2}],
    }]
    preds, refs, tok = build_items(
        rows, silence_token="<w>", eos_tokens=["<e>"],
        default_chunks_per_second=5.0
    )
    s = StreamingMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert s["n_content_match"] == 1
    assert s["n_false_alarm"] == 1  # 'wrong' beyond the horizon still scored


def test_no_hits_produces_strict_json(tmp_path):
    """A sample with no hits yields NaN delay fields; outputs must be valid
    strict JSON (null, not the bare NaN token) with stable CSV columns."""
    src = tmp_path / "miss.jsonl"
    src.write_text(json.dumps({
        "sample_id": "miss", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>"] * 6,
        "emissions": [{"text": "dog", "t_start": 0.5, "t_end": 0.7}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>", "--chunks-per-second", "5.0",
    ])
    assert rc == 0
    text = (out / "all_results.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    results = json.loads(
        text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))
    )
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    assert sample["delay_mean"] is None
    assert sample["delays"] == []
    ds = results["dataset_level_metrics"]["streaming_metrics"]
    assert ds["n_delay"] == 0
    assert ds["delay_mean"] is None
    # CSV keeps the delay column (empty cell), stable header across runs
    csv_lines = (out / "summary.csv").read_text().splitlines()
    assert "delay_mean" in csv_lines[0]


def test_emission_with_missing_t_start_raises():
    """Emissions must carry finite, ordered t_start/t_end — no silent -inf
    windows (streaming) or auto-miss (post-audio) for malformed input."""
    rows = [{
        "sample_id": "bad-em", "prefix_len": 0, "chunks_per_second": 5.0,
        "pred": ["<w>", "hi", "<w>"],
        "emissions": [{"text": "hi", "t_end": 0.3}],  # t_start missing
    }]
    with pytest.raises(ValueError, match="finite t_start <= t_end"):
        build_items(rows, silence_token="<w>", eos_tokens=["<e>"],
                    default_chunks_per_second=5.0)

    rows[0]["emissions"] = [{"text": "hi", "t_start": 0.5, "t_end": 0.3}]
    with pytest.raises(ValueError, match="finite t_start <= t_end"):
        build_items(rows, silence_token="<w>", eos_tokens=["<e>"],
                    default_chunks_per_second=5.0)


def test_grace_window_sec_overrides_row_chunk_rate(tmp_path):
    """An explicit --grace-window-sec pins the protocol regardless of the
    row's chunks_per_second (transport chunk size must not move the
    acceptance windows)."""
    src = tmp_path / "pin.jsonl"
    # Late answer at 1.0 vs ref [0.2, 0.4]: hits only when 2C >= 0.8.
    src.write_text(json.dumps({
        "sample_id": "pin", "prefix_len": 0, "chunks_per_second": 6.25,
        "pred": ["<w>"] * 5 + ["dog"] + ["<w>"] * 3,
        "pred_runs": [{"text": "dog", "t_start": 1.0, "t_end": 1.1}],
        "emissions": [{"text": "dog", "t_start": 0.2, "t_end": 0.4}],
    }) + "\n")
    out = tmp_path / "out"
    rc = main([
        str(src), "--output-dir", str(out),
        "--silence-token", "<w>", "--eos-token", "<e>",
        "--grace-window-sec", "0.4",
    ])
    assert rc == 0
    results = json.loads((out / "all_results.json").read_text())
    assert results["config"]["grace_window_sec"] == 0.4
    sample = results["sample_level_metrics"]["streaming_metrics"][0]
    # 6.25 chunks/s would give 0.16/0.32; the override pins 0.4/0.8.
    assert sample["grace_windows"] == [
        ["during_word", 0.0], ["plus1c", 0.4], ["plus2c", 0.8],
    ]
    levels = sample["emissions"][0]["levels"]
    assert levels["plus1c"]["fuzzy"]["hit"] is False
    assert levels["plus2c"]["fuzzy"]["hit"] is True
