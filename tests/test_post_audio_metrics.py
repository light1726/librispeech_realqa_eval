"""Tests for post-audio ordered exclusive metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realqa.eval import build_items, main
from realqa.post_audio_metrics import PostAudioMetrics


def _items(rows):
    return build_items(
        rows,
        silence_token="<w>",
        eos_tokens=["<e>"],
        default_chunks_per_second=5.0,
    )


def test_one_blob_matches_at_most_one_target():
    """Intact blob 'one two' is one unit → exclusive → only first matching GT."""
    # chunks_per_second=5 → onset at index 10 → 2.0s
    rows = [{
        "sample_id": "blob",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 12,
        "pred": ["<w>"] * 10 + ["one two"] + ["<w>"],
        "emissions": [
            {"text": "one", "t_start": 0.5, "t_end": 0.8},
            {"text": "two", "t_start": 1.0, "t_end": 1.3},
        ],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["n_tp_fuzzy"] == 1
    assert sample["n_missed_fuzzy"] == 1
    assert sample["n_false_alarm_fuzzy"] == 0
    assert abs(sample["delays_fuzzy"][0] - (2.0 - 0.5)) < 1e-6


def test_two_runs_match_two_targets_in_order():
    """Separate non-wait runs align 1–1 in order."""
    rows = [{
        "sample_id": "two_runs",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 8,
        "pred": ["<w>", "<w>", "one", "<w>", "two", "<w>", "<w>", "<w>"],
        "emissions": [
            {"text": "one", "t_start": 0.2, "t_end": 0.4},
            {"text": "two", "t_start": 0.6, "t_end": 0.8},
        ],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["n_tp_fuzzy"] == 2
    assert sample["n_missed_fuzzy"] == 0
    assert sample["n_false_alarm_fuzzy"] == 0
    assert abs(sample["delays_fuzzy"][0] - (0.4 - 0.2)) < 1e-6  # onset frame 2
    assert abs(sample["delays_fuzzy"][1] - (0.8 - 0.6)) < 1e-6  # onset frame 4


def test_wrong_order_reduces_hits():
    """Pred runs two then one vs refs one→two: second run can match one; two misses."""
    rows = [{
        "sample_id": "reorder",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 6,
        "pred": ["<w>", "two", "<w>", "one", "<w>", "<w>"],
        "emissions": [
            {"text": "one", "t_start": 0.0, "t_end": 0.2},
            {"text": "two", "t_start": 0.4, "t_end": 0.6},
        ],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    # ref one skips 'two', matches 'one'; then ref two has no remaining unit
    assert sample["n_tp_fuzzy"] == 1
    assert sample["n_missed_fuzzy"] == 1
    assert sample["n_false_alarm_fuzzy"] == 1  # leftover 'two'


def test_early_onset_rejected():
    """Pred before ref_t_start - eps cannot match that target."""
    rows = [{
        "sample_id": "early",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 10,
        "pred": ["hi"] + ["<w>"] * 9,
        "emissions": [{"text": "hi", "t_start": 1.0, "t_end": 1.2}],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics(early_eps_sec=0.05).compute_batch(
        preds, refs, tokenizer=tok,
    )[0]
    assert sample["n_tp_fuzzy"] == 0
    assert sample["n_missed_fuzzy"] == 1
    assert sample["n_false_alarm_fuzzy"] == 1


def test_extra_pred_run_is_fa():
    rows = [{
        "sample_id": "extra",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 8,
        "pred": ["<w>", "one", "<w>", "junk", "<w>", "<w>", "<w>", "<w>"],
        "emissions": [{"text": "one", "t_start": 0.1, "t_end": 0.3}],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["n_tp_fuzzy"] == 1
    assert sample["n_false_alarm_fuzzy"] == 1


def test_long_blob_containing_answer_is_one_unit():
    """Conversational blob counts as one FA or one TP, not per-token FAs."""
    rows = [{
        "sample_id": "long",
        "prefix_len": 0,
        "chunks_per_second": 5.0,
        "timeline_len": 6,
        "pred": ["<w>"] * 3 + ["said our learned guide"] + ["<w>", "<w>"],
        "emissions": [{"text": "guide", "t_start": 0.2, "t_end": 0.4}],
    }]
    preds, refs, tok = _items(rows)
    sample = PostAudioMetrics().compute_batch(preds, refs, tokenizer=tok)[0]
    assert sample["n_tp_fuzzy"] == 1
    assert sample["n_false_alarm_fuzzy"] == 0
    assert sample["total_pred_units"] == 1


def test_cli_post_audio_mode(tmp_path):
    demo = ROOT / "examples" / "demo.jsonl"
    out = tmp_path / "out"
    rc = main([
        str(demo),
        "--output-dir", str(out),
        "--silence-token", "<silence>",
        "--eos-token", "<|im_end|>",
        "--chunks-per-second", "6.25",
        "--metric-mode", "post_audio",
    ])
    assert rc == 0
    import json
    results = json.loads((out / "all_results.json").read_text())
    assert results["task_type"] == "POST_AUDIO"
    assert "post_audio_metrics" in results["dataset_level_metrics"]
    m = results["dataset_level_metrics"]["post_audio_metrics"]
    assert "delay_mean" in m
    assert "emission_f1_fuzzy" in m
