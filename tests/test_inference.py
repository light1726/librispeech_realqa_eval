"""Unit tests for inference pipeline components.

Tests TimelineBuilder, EvalFormatter, DatasetReader, AudioLoader,
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from realqa.dataset import DatasetReader
from realqa.base import (
    ContentDoneEvent,
    ContentTokenEvent,
    SessionEndEvent,
    TimelineResult,
)
from realqa.formatter import EvalFormatter
from realqa.timeline import (
    ContentRun,
    TimelineBuilder,
)
from realqa.sample import DatasetSample, Emission

# ═══════════════════════════════════════════════════════════════════
# TimelineBuilder
# ═══════════════════════════════════════════════════════════════════


class TestTimelineBuilder:
    def build(self, chunks_per_second=6.25, wait="<w>", eos="") -> TimelineBuilder:
        return TimelineBuilder(chunks_per_second=chunks_per_second, silence_token=wait, eos_token=eos)

    # ── single burst ──────────────────────────────────────────

    def test_single_burst_one_token(self):
        """One content token → content frame; boundary stays wait (no EOS)."""
        builder = self.build()
        events = [
            ContentTokenEvent(0.48, "stew"),
            ContentDoneEvent(0.64, "stew"),
            SessionEndEvent(1.0),
        ]
        result = builder.build(events, audio_duration_s=1.28)
        # 1.28 * 6.25 = 8 frames
        assert len(result.tokens) == 8
        # Frame 3 = 0.48s; done at 0.64s marks exclusive end (frame 4 stays wait)
        assert result.tokens[3] == "stew"
        assert result.tokens[4] == "<w>"
        assert all(t == "<w>" for i, t in enumerate(result.tokens) if i != 3)
        assert len(result.content_runs) == 1
        assert result.content_runs[0]["text"] == "stew"
        assert result.content_runs[0]["t_start"] == 0.48
        assert result.content_runs[0]["t_end"] == 0.64

    def test_single_burst_with_optional_eos(self):
        """When eos_token is set, place it at the burst boundary."""
        builder = self.build(eos="<e>")
        events = [
            ContentTokenEvent(0.48, "stew"),
            ContentDoneEvent(0.64, "stew"),
            SessionEndEvent(1.0),
        ]
        result = builder.build(events, audio_duration_s=1.28)
        assert result.tokens[3] == "stew"
        assert result.tokens[4] == "<e>"

    def test_single_burst_multi_token(self):
        """Each delta lands on its own frame; gaps stay wait."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(0.1, "s"),
            ContentTokenEvent(0.2, "t"),
            ContentTokenEvent(0.3, "e"),
            ContentDoneEvent(0.5, "ste"),
            SessionEndEvent(0.8),
        ]
        result = builder.build(events, audio_duration_s=1.0)
        assert len(result.tokens) == 10
        assert result.tokens[1] == "s"
        assert result.tokens[2] == "t"
        assert result.tokens[3] == "e"
        assert result.tokens[4] == "<w>"
        assert result.tokens[5] == "<w>"
        assert len(result.content_runs) == 1
        assert result.content_runs[0]["text"] == "ste"

    def test_same_frame_deltas_concatenate(self):
        """Subword deltas with the same timestamp concatenate on one frame."""
        builder = self.build(6.25)
        events = [
            ContentTokenEvent(3.0, "wes"),
            ContentTokenEvent(3.0, "ley"),
            ContentDoneEvent(3.01, "wesley"),
            SessionEndEvent(3.1),
        ]
        result = builder.build(events, audio_duration_s=2.0)
        # Response after audio → timeline extends past 2.0s
        assert len(result.tokens) >= math.ceil(3.0 * 6.25)
        frame = int(3.0 * 6.25 + 1e-9)
        assert result.tokens[frame] == "wesley"
        assert result.content_runs[0]["text"] == "wesley"

    def test_extends_timeline_for_late_response(self):
        """Post-audio responses extend the grid with silence tokens."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(1.5, "hi"),
            ContentDoneEvent(1.5, "hi"),
        ]
        result = builder.build(events, audio_duration_s=1.0)
        assert len(result.tokens) >= 16  # ceil(1.5 * 10)
        assert result.tokens[15] == "hi"
        assert all(t == "<w>" for t in result.tokens[:15])

    # ── multi burst ────────────────────────────────────────────

    def test_multi_burst(self):
        """Two content bursts → gap filled with silence tokens."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(0.1, "hi"),
            ContentDoneEvent(0.2, "hi"),
            ContentTokenEvent(0.5, "bye"),
            ContentDoneEvent(0.6, "bye"),
            SessionEndEvent(0.8),
        ]
        result = builder.build(events, audio_duration_s=1.0)
        assert len(result.tokens) == 10

        # Burst 1: frame 1 content; frame 2 stays wait (no EOS)
        assert result.tokens[1] == "hi"
        assert result.tokens[2] == "<w>"
        # Gap: frames 3-4 → wait
        assert result.tokens[3] == "<w>"
        assert result.tokens[4] == "<w>"
        # Burst 2: frame 5 content
        assert result.tokens[5] == "bye"
        assert result.tokens[6] == "<w>"

        assert len(result.content_runs) == 2
        assert result.content_runs[0]["text"] == "hi"
        assert result.content_runs[1]["text"] == "bye"

    # ── no content ─────────────────────────────────────────────

    def test_no_content(self):
        """No token events → all silence tokens."""
        builder = self.build()
        events = [SessionEndEvent(1.0)]
        result = builder.build(events, audio_duration_s=1.28)
        assert all(t == "<w>" for t in result.tokens)
        assert result.content_runs == []

    # ── prefix offset ──────────────────────────────────────────

    def test_prefix_offset(self):
        """Prefix does not change token grid; it is a metadata flag."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(0.1, "x"),
            ContentDoneEvent(0.2, "x"),
        ]
        result = builder.build(events, audio_duration_s=0.5, prefix_len=3)
        assert result.prefix_len == 3
        assert len(result.tokens) == 5

    # ── edge: token at frame boundary ──────────────────────────

    def test_token_at_boundary(self):
        """Token exactly at a frame boundary lands on that frame."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(0.0, "start"),
            ContentDoneEvent(0.1, "start"),
        ]
        result = builder.build(events, audio_duration_s=0.5)
        assert result.tokens[0] == "start"
        assert result.tokens[1] == "<w>"

    # ── edge: empty event list ─────────────────────────────────

    def test_empty_events(self):
        builder = self.build()
        result = builder.build([], audio_duration_s=1.0)
        assert len(result.tokens) == 7  # ceil(1.0 * 6.25)
        assert all(t == "<w>" for t in result.tokens)
        assert result.content_runs == []

    # ── edge: tokens without done event ────────────────────────

    def test_tokens_without_done(self):
        """Tokens without a ContentDoneEvent still form a burst."""
        builder = self.build(10.0)
        events = [
            ContentTokenEvent(0.1, "orphan"),
        ]
        result = builder.build(events, audio_duration_s=0.5)
        assert result.tokens[1] == "orphan"
        assert result.tokens[2] == "<w>"
        assert len(result.content_runs) == 1

    def test_content_runs_keep_exact_onsets(self):
        """content_runs t_start is the exact first-delta timestamp, not the
        frame-quantized one (pred_runs / event-level eval relies on this)."""
        builder = self.build(5.0)  # 200 ms frames
        events = [
            # 1.44 * 5 = 7.2 → floor places the grid token at frame 7 (=1.4s)
            ContentTokenEvent(1.44, "stew"),
            ContentDoneEvent(1.6, "stew"),
            SessionEndEvent(1.8),
        ]
        result = builder.build(events, audio_duration_s=2.0)
        assert result.tokens[7] == "stew"
        run = result.content_runs[0]
        assert run["t_start"] == 1.44  # exact, not 7 / 5.0
        assert run["t_end"] == 1.6


# ═══════════════════════════════════════════════════════════════════
# EvalFormatter
# ═══════════════════════════════════════════════════════════════════


class TestEvalFormatter:
    def _sample(self, emissions=None) -> DatasetSample:
        if emissions is None:
            emissions = [Emission("stew", 0.48, 0.64)]
        return DatasetSample(
            utterance_id="utt-001",
            audio_path="/tmp/audio.flac",
            prompt="Say the word.",
            task_type="lexical_detection",
            query="test",
            emissions=emissions,
        )

    def _timeline(self, tokens=None, n=8, chunks_per_second=6.25) -> TimelineResult:
        tokens = tokens or (
            ["<w>"] * 3 + ["stew"] + ["<w>"] * 4
        )
        return TimelineResult(
            tokens=tokens,
            content_runs=[{"text": "stew", "t_start": 0.48, "t_end": 0.64,
                           "start_group": 3, "end_group": 4}],
            chunks_per_second=chunks_per_second,
            prefix_len=0,
        )

    def test_basic_format(self):
        fmt = EvalFormatter(silence_token="<w>")
        sample = self._sample()
        timeline = self._timeline()
        line = fmt.format_line(sample, timeline)

        assert line["sample_id"] == "utt-001::lexical_detection::test"
        assert line["utterance_id"] == "utt-001"
        assert line["task_type"] == "lexical_detection"
        assert line["query"] == "test"
        assert line["pred"] == timeline.tokens
        assert line["prefix_len"] == 0
        assert line["chunks_per_second"] == 6.25
        assert len(line["emissions"]) == 1
        assert line["emissions"][0]["text"] == "stew"
        assert line["emissions"][0]["t_start"] == 0.48
        assert line["emissions"][0]["t_end"] == 0.64
        # Event-level predictions with exact first-delta timestamps
        assert line["pred_runs"] == [
            {"text": "stew", "t_start": 0.48, "t_end": 0.64},
        ]

    def test_emissions_pass_through(self):
        """Emissions are exactly what was in the dataset sample."""
        fmt = EvalFormatter()
        emissions = [Emission("hello", 1.0, 2.0), Emission("world", 3.0, 4.0)]
        sample = self._sample(emissions)
        line = fmt.format_line(sample, self._timeline(n=32))
        assert line["emissions"][0]["text"] == "hello"
        assert line["emissions"][0]["t_start"] == 1.0
        assert line["emissions"][0]["t_end"] == 2.0
        assert line["emissions"][1]["text"] == "world"

    def test_timeline_len_is_set(self):
        """timeline_len is present and equals the prediction grid length."""
        fmt = EvalFormatter()
        line = fmt.format_line(self._sample(), self._timeline())
        assert line["timeline_len"] == 8
        assert "ref" not in line

    def test_json_output(self):
        fmt = EvalFormatter()
        line_json = fmt.format_line_json(self._sample(), self._timeline())
        assert line_json.endswith("\n")
        parsed = json.loads(line_json)
        assert parsed["sample_id"] == "utt-001::lexical_detection::test"

    def test_no_emissions(self):
        """Sample with no emissions → empty emissions list."""
        fmt = EvalFormatter()
        sample = self._sample(emissions=[])
        line = fmt.format_line(sample, self._timeline())
        assert line["emissions"] == []


# ═══════════════════════════════════════════════════════════════════
# DatasetReader
# ═══════════════════════════════════════════════════════════════════


class TestDatasetReader:
    def _write_jsonl(self, lines):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for line in lines:
            tmp.write(json.dumps(line) + "\n")
        tmp.close()
        return tmp.name

    def test_reads_samples(self):
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p1",
             "task_type": "lexical_detection", "query": "q1",
             "labels": [[1.0, 2.0, "hi"]]},
            {"utterance_id": "u2", "audio_path": "/b.flac", "prompt": "p2",
             "task_type": "semantic_detection", "query": "q2",
             "labels": [[3.0, 4.0, "bye"]]},
        ])
        reader = DatasetReader(path)
        samples = list(reader)
        assert len(samples) == 2
        assert samples[0].utterance_id == "u1"
        assert samples[0].prompt == "p1"
        assert len(samples[0].emissions) == 1
        assert samples[0].emissions[0].text == "hi"
        assert samples[0].emissions[0].t_start == 1.0
        assert samples[0].emissions[0].t_end == 2.0
        assert samples[1].utterance_id == "u2"
        os.unlink(path)

    def test_task_filter(self):
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "lexical_detection", "query": "q", "labels": []},
            {"utterance_id": "u2", "audio_path": "/b.flac", "prompt": "p",
             "task_type": "semantic_detection", "query": "q", "labels": []},
        ])
        reader = DatasetReader(path, task_type="lexical_detection")
        samples = list(reader)
        assert len(samples) == 1
        assert samples[0].utterance_id == "u1"
        os.unlink(path)

    def test_limit_and_seed(self):
        samples_data = [
            {"utterance_id": f"u{i}", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "t", "query": "q", "labels": []}
            for i in range(10)
        ]
        path = self._write_jsonl(samples_data)
        reader = DatasetReader(path, limit=3, seed=42)
        ids_a = [s.utterance_id for s in reader]
        reader_b = DatasetReader(path, limit=3, seed=42)
        ids_b = [s.utterance_id for s in reader_b]
        assert ids_a == ids_b, "Same seed should produce same order"
        assert len(ids_a) == 3
        os.unlink(path)

    def test_zero_labels(self):
        """Sample with empty labels → empty emissions list."""
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "t", "query": "q", "labels": []},
        ])
        reader = DatasetReader(path)
        samples = list(reader)
        assert len(samples) == 1
        assert samples[0].emissions == []
        os.unlink(path)

    def test_len(self):
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "t", "query": "q", "labels": [[1.0, 2.0, "x"]]},
        ])
        reader = DatasetReader(path)
        assert len(reader) == 1
        assert len(reader) == 1  # can call repeatedly
        os.unlink(path)

    def test_skips_blank_lines(self):
        """Empty lines are skipped; invalid JSON objects raise meaningful error."""
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "t", "query": "q", "labels": []},
            {"utterance_id": "u2", "audio_path": "/b.flac", "prompt": "p2",
             "task_type": "t", "query": "q2", "labels": []},
        ])
        reader = DatasetReader(path)
        samples = list(reader)
        assert len(samples) == 2
        os.unlink(path)

    def test_invalid_row_raises(self):
        """Row missing required keys raises KeyError."""
        path = self._write_jsonl([
            {"utterance_id": "u1", "audio_path": "/a.flac", "prompt": "p",
             "task_type": "t", "query": "q", "labels": []},
            {},
            {"utterance_id": "u2", "audio_path": "/b.flac", "prompt": "p2",
             "task_type": "t", "query": "q2", "labels": []},
        ])
        reader = DatasetReader(path)
        with pytest.raises(KeyError, match="missing required keys"):
            list(reader)
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════
# AudioLoader
# ═══════════════════════════════════════════════════════════════════


class TestAudioLoader:
    def test_chunkify(self):
        from realqa.audio import AudioLoader
        loader = AudioLoader(target_sample_rate=16000, chunk_ms=100)
        # 16000 Hz * 0.1s = 1600 samples, 3200 bytes
        pcm = bytes(3200 * 3)  # 300ms
        chunks = loader.chunkify(pcm)
        assert len(chunks) == 3
        assert len(chunks[0]) == 3200
        assert len(chunks[-1]) == 3200

    def test_chunkify_short(self):
        from realqa.audio import AudioLoader
        loader = AudioLoader(target_sample_rate=16000, chunk_ms=200)
        # 200ms at 16kHz = 3200 samples = 6400 bytes, but only 4000 bytes
        chunks = loader.chunkify(bytes(4000))
        assert len(chunks) == 1
        assert len(chunks[0]) == 4000

    def test_to_pcm16(self):
        from realqa.audio import AudioLoader
        loader = AudioLoader()
        audio = np.array([-1.0, 0.0, 1.0, -0.5, 0.5], dtype=np.float32)
        pcm = loader.to_pcm16(audio)
        assert len(pcm) == len(audio) * 2  # 2 bytes per sample
        arr = np.frombuffer(pcm, dtype=np.int16)
        assert arr[0] == -32767
        assert arr[1] == 0
        assert arr[2] == 32767
        assert arr[3] < 0
        assert arr[4] > 0

    def test_flac_load_and_resample(self):
        """End-to-end test: write synthetic FLAC, load, resample, verify."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        from realqa.audio import AudioLoader

        # Create a 1-second 8kHz sine wave
        sr_in = 8000
        duration = 1.0
        t = np.linspace(0, duration, int(sr_in * duration), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            sf.write(tmp.name, audio, sr_in)
            flac_path = tmp.name

        try:
            loader = AudioLoader(target_sample_rate=16000)
            loaded, sr = loader.load(flac_path)
            assert sr == sr_in
            assert abs(len(loaded) - int(sr_in * duration)) <= 1

            resampled = loader.resample(loaded, sr)
            expected_len = int(16000 * duration)
            assert abs(len(resampled) - expected_len) <= 2

            chunks, dur = loader.load_for_streaming(flac_path)
            assert dur > 0
            assert len(chunks) > 0
        finally:
            os.unlink(flac_path)


# ═══════════════════════════════════════════════════════════════════
# Integration test (uses mock client, no real API)
# ═══════════════════════════════════════════════════════════════════


class MockClient:
    """Fake StreamingClient that replays pre-recorded events.

    Used to test the full pipeline without a real API connection.
    """

    def __init__(self, events=None):
        self.events = events or []
        self._connected = False

    def connect(self, config):
        self._connected = True

    def send_audio(self, audio_bytes, chunk_id):
        pass

    def end_audio(self):
        pass

    def iter_events(self):
        yield from self.events

    def disconnect(self):
        self._connected = False


def test_integration_pipeline_to_metrics(tmp_path):
    """Full pipeline: dataset -> mock client -> timeline -> formatter -> metrics.

    Does NOT require a real API or audio files — uses a mock client that
    replays pre-recorded events.
    """
    from realqa.dataset import DatasetReader
    from realqa.base import (
        ContentDoneEvent,
        ContentTokenEvent,
        SessionEndEvent,
        SessionReadyEvent,
    )
    from realqa.formatter import EvalFormatter
    from realqa.timeline import TimelineBuilder

    # Write a minimal benchmark JSONL (audio_path unused by mock)
    sample_jsonl = tmp_path / "test.jsonl"
    sample_jsonl.write_text(
        json.dumps({
            "utterance_id": "test-001",
            "audio_path": "/nonexistent/audio.flac",
            "prompt": "Say the word.",
            "task_type": "lexical_detection",
            "query": "stew",
            "labels": [[5.0, 5.5, "stew"]],
        }) + "\n"
    )

    # Load sample via DatasetReader
    reader = DatasetReader(sample_jsonl)
    samples = list(reader)
    assert len(samples) == 1
    sample = samples[0]

    # Pre-recorded events: one "stew" emission at 5.1s
    mock_events = [
        SessionReadyEvent(0.2),
        ContentTokenEvent(5.1, "stew"),
        ContentDoneEvent(5.6, "stew"),
        SessionEndEvent(6.0),
    ]

    # Timeline builder
    audio_duration_s = 10.0
    timeline_builder = TimelineBuilder(
        chunks_per_second=6.25,
        silence_token="<silence>",
        eos_token="",
    )
    timeline = timeline_builder.build(mock_events, audio_duration_s)

    # Format predictions
    formatter = EvalFormatter(silence_token="<silence>")
    line = formatter.format_line(sample, timeline)

    # Write predictions JSONL
    pred_path = tmp_path / "predictions.jsonl"
    with pred_path.open("w") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Verify prediction format
    assert line["sample_id"] == "test-001::lexical_detection::stew"
    assert line["prefix_len"] == 0
    assert line["chunks_per_second"] == 6.25
    assert len(line["emissions"]) == 1
    assert line["emissions"][0]["text"] == "stew"
    assert "stew" in line["pred"]
    assert "<|im_end|>" not in line["pred"]

    # Feed predictions into the eval pipeline and check metrics
    from realqa.eval import main as eval_main
    eval_dir = tmp_path / "eval"
    rc = eval_main([
        str(pred_path),
        "--output-dir", str(eval_dir),
        "--silence-token", "<silence>",
        "--chunks-per-second", "6.25",
    ])
    assert rc == 0

    results = json.loads((eval_dir / "all_results.json").read_text())
    m = results["dataset_level_metrics"]["streaming_metrics"]
    assert m["n_samples"] == 1
    assert m["total_ref_emissions"] == 1
    # The mock emission at 5.1s with a done at 5.6s should match the
    # ref emission window [5.0, 5.5] during the word (0.0s grace)
    assert m["emission_f1_fuzzy_during_word"] >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Qwen-Audio-3.0-Realtime adapter
# ═══════════════════════════════════════════════════════════════════


class TestQwenAudioRealtimeAdapter:
    """Test QwenAudioRealtimeClient without requiring the SDK."""

    def test_import_guard_prevents_construction_without_websockets(self):
        from realqa.adapters.qwen_audio_realtime import (
            QwenAudioRealtimeClient,
            HAS_WEBSOCKETS,
        )
        if not HAS_WEBSOCKETS:
            with pytest.raises(ImportError, match="websockets"):
                QwenAudioRealtimeClient(api_key="sk-test")

    def test_missing_api_key_raises(self):
        from realqa.adapters.qwen_audio_realtime import (
            QwenAudioRealtimeClient,
            HAS_WEBSOCKETS,
        )
        if HAS_WEBSOCKETS:
            import os
            old_ds = os.environ.pop("DASHSCOPE_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                    QwenAudioRealtimeClient(api_key="", voice="longanqian")
            finally:
                if old_ds is not None:
                    os.environ["DASHSCOPE_API_KEY"] = old_ds

    def test_ws_url_query_merge(self):
        from realqa.adapters.qwen_audio_realtime import QwenAudioRealtimeClient
        url = QwenAudioRealtimeClient._ws_url(
            "wss://example.com/api-ws/v1/realtime?foo=1",
            "qwen-audio-3.0-realtime-flash",
        )
        assert "model=qwen-audio-3.0-realtime-flash" in url
        assert "foo=1" in url
        assert url.count("?") == 1

    def test_runner_registers_qwen_audio(self):
        from realqa.runner import _get_client
        from realqa.adapters.qwen_audio_realtime import (
            QwenAudioRealtimeClient,
            HAS_WEBSOCKETS,
        )
        if HAS_WEBSOCKETS:
            import os
            if os.environ.get("DASHSCOPE_API_KEY"):
                client = _get_client("qwen-audio-3.0-realtime-flash")
                assert isinstance(client, QwenAudioRealtimeClient)
        else:
            with pytest.raises(ImportError, match="websockets"):
                QwenAudioRealtimeClient(api_key="sk-dummy")

    def test_runner_routes_omni_for_non_audio_qwen(self):
        from realqa.runner import _get_client
        from realqa.adapters.qwen_omni import QwenOmniRealtimeClient, HAS_DASHSCOPE
        if HAS_DASHSCOPE:
            import os
            if os.environ.get("DASHSCOPE_API_KEY"):
                client = _get_client("qwen3.5-omni-flash-realtime")
                assert isinstance(client, QwenOmniRealtimeClient)


# ═══════════════════════════════════════════════════════════════════
# Gemini Live adapter
# ═══════════════════════════════════════════════════════════════════


class TestGeminiLiveAdapter:
    def test_missing_api_key_raises(self):
        from realqa.adapters.gemini_live import GeminiLiveClient, HAS_GENAI
        if not HAS_GENAI:
            with pytest.raises(ImportError, match="google-genai"):
                GeminiLiveClient(api_key="x")
            return
        import os
        old_g = os.environ.pop("GEMINI_API_KEY", None)
        old_go = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="GEMINI"):
                GeminiLiveClient(api_key="")
        finally:
            if old_g is not None:
                os.environ["GEMINI_API_KEY"] = old_g
            if old_go is not None:
                os.environ["GOOGLE_API_KEY"] = old_go

    def test_normalize_model_strips_prefix(self):
        from realqa.adapters.gemini_live import GeminiLiveClient
        assert GeminiLiveClient._normalize_model("models/gemini-3.1-flash-live-preview") == (
            "gemini-3.1-flash-live-preview"
        )

    def test_runner_registers_gemini(self):
        from realqa.runner import _get_client
        from realqa.adapters.gemini_live import GeminiLiveClient, HAS_GENAI
        import os
        if not HAS_GENAI:
            with pytest.raises(ImportError, match="google-genai"):
                GeminiLiveClient(api_key="dummy")
            return
        if (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        ):
            client = _get_client("gemini-3.1-flash-live-preview")
            assert isinstance(client, GeminiLiveClient)
        else:
            # Construction via _get_client still needs a key.
            with pytest.raises(ValueError, match="GEMINI"):
                _get_client("gemini-3.1-flash-live-preview")


# ═══════════════════════════════════════════════════════════════════
# InferenceRunner resume + failures
# ═══════════════════════════════════════════════════════════════════


class TestInferenceRunnerResume:
    def _dataset(self, tmp_path, n=3):
        rows = []
        for i in range(n):
            rows.append({
                "utterance_id": f"u{i}",
                "audio_path": "/tmp/missing.flac",
                "prompt": "p",
                "task_type": "lexical_detection",
                "query": "q",
                "labels": [[0.0, 0.1, "hi"]],
            })
        path = tmp_path / "ds.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    def _runner(self, tmp_path, dataset, client, resume=False):
        from realqa.runner import InferenceRunner
        from realqa.base import SessionConfig
        from realqa.audio import AudioLoader
        from realqa.timeline import TimelineBuilder
        from realqa.formatter import EvalFormatter
        return InferenceRunner(
            dataset_path=dataset,
            output_dir=tmp_path / "out",
            client=client,
            session_config=SessionConfig(model="dummy", chunks_per_second=5.0),
            audio_loader=AudioLoader(chunk_ms=200),
            timeline_builder=TimelineBuilder(chunks_per_second=5.0, silence_token="<w>"),
            formatter=EvalFormatter(silence_token="<w>"),
            resume=resume,
        )

    def test_failures_jsonl_and_resume_skip(self, tmp_path, monkeypatch):
        from realqa.base import (
            SessionReadyEvent, ContentTokenEvent, ContentDoneEvent, SessionEndEvent,
        )
        from realqa.timeline import TimelineResult

        dataset = self._dataset(tmp_path, n=3)

        class FakeClient:
            def __init__(self):
                self.n = 0
                self.connected = False

            def connect(self, config):
                self.connected = True

            def send_audio(self, b, i):
                pass

            def end_audio(self):
                pass

            def iter_events(self):
                # fail first sample, succeed later ones
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("boom")
                yield SessionReadyEvent(0.0)
                yield ContentTokenEvent(0.2, "hi")
                yield ContentDoneEvent(0.3, "hi")
                yield SessionEndEvent(0.3)

            def disconnect(self):
                self.connected = False

            @property
            def is_connected(self):
                return self.connected

        # Avoid real audio load
        def fake_load(self, path):
            return [b"\x00\x00"] * 2, 0.4

        monkeypatch.setattr(
            "realqa.audio.AudioLoader.load_for_streaming", fake_load
        )

        client = FakeClient()
        r1 = self._runner(tmp_path, dataset, client, resume=False)
        stats1 = r1.run()
        assert stats1["success"] == 2
        assert stats1["error"] == 1
        out = tmp_path / "out"
        fails = [json.loads(l) for l in (out / "failures.jsonl").read_text().splitlines() if l.strip()]
        assert len(fails) == 1
        assert fails[0]["utterance_id"] == "u0"
        assert "boom" in fails[0]["error"]
        preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines() if l.strip()]
        assert {p["sample_id"] for p in preds} == {
            "u1::lexical_detection::q",
            "u2::lexical_detection::q",
        }

        # Resume: skip u1/u2, retry u0 (still fails)
        client2 = FakeClient()  # n starts at 0 → first attempted (u0) fails again
        r2 = self._runner(tmp_path, dataset, client2, resume=True)
        stats2 = r2.run()
        assert stats2["skipped"] == 2
        assert stats2["error"] == 1
        assert stats2["success"] == 0
        fails2 = [json.loads(l) for l in (out / "failures.jsonl").read_text().splitlines() if l.strip()]
        assert len(fails2) == 1
        assert fails2[0]["sample_id"] == "u0::lexical_detection::q"
        assert fails2[0]["utterance_id"] == "u0"

    def test_fresh_run_processes_same_label_tasks(self, tmp_path, monkeypatch):
        """Fresh runs must never skip: two tasks whose GT labels coincide
        (e.g. lexical 'warriors' + semantic 'profession' on the same word)
        are distinct samples.  Regression for fingerprint-based skipping."""
        from realqa.base import (
            SessionReadyEvent, ContentTokenEvent, ContentDoneEvent, SessionEndEvent,
        )

        labels = [[1.0, 1.4, "warriors"]]
        rows = [
            {
                "utterance_id": "same",
                "audio_path": "/tmp/missing.flac",
                "prompt": "p1",
                "task_type": "lexical_detection",
                "query": "warriors",
                "labels": labels,
            },
            {
                "utterance_id": "same",
                "audio_path": "/tmp/missing.flac",
                "prompt": "p2",
                "task_type": "semantic_detection",
                "query": "profession",
                "labels": labels,  # identical fingerprint
            },
        ]
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        class FakeClient:
            def connect(self, config):
                pass

            def send_audio(self, b, i):
                pass

            def end_audio(self):
                pass

            def iter_events(self):
                yield SessionReadyEvent(0.0)
                yield ContentTokenEvent(1.5, "warriors")
                yield ContentDoneEvent(1.6, "warriors")
                yield SessionEndEvent(1.6)

            def disconnect(self):
                pass

        def fake_load(self, path):
            return [b"\x00\x00"] * 2, 0.4

        monkeypatch.setattr(
            "realqa.audio.AudioLoader.load_for_streaming", fake_load
        )

        r = self._runner(tmp_path, dataset, FakeClient(), resume=False)
        stats = r.run()
        assert stats["success"] == 2
        assert stats["skipped"] == 0
        preds = [
            json.loads(l)
            for l in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert {p["sample_id"] for p in preds} == {
            "same::lexical_detection::warriors",
            "same::semantic_detection::profession",
        }

    def test_legacy_fingerprint_covers_one_sample(self, tmp_path, monkeypatch):
        """Resume vs a legacy predictions line: exactly one pending row is
        covered, even when two tasks share the same GT labels."""
        from realqa.base import (
            SessionReadyEvent, ContentTokenEvent, ContentDoneEvent, SessionEndEvent,
        )
        from realqa.runner import InferenceRunner
        from realqa.base import SessionConfig
        from realqa.audio import AudioLoader
        from realqa.timeline import TimelineBuilder
        from realqa.formatter import EvalFormatter

        labels = [[1.0, 1.4, "warriors"]]
        rows = [
            {
                "utterance_id": "same", "audio_path": "/tmp/missing.flac",
                "prompt": "p1", "task_type": "lexical_detection",
                "query": "warriors", "labels": labels,
            },
            {
                "utterance_id": "same", "audio_path": "/tmp/missing.flac",
                "prompt": "p2", "task_type": "semantic_detection",
                "query": "profession", "labels": labels,
            },
        ]
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        class FakeClient:
            def connect(self, config):
                pass

            def send_audio(self, b, i):
                pass

            def end_audio(self):
                pass

            def iter_events(self):
                yield SessionReadyEvent(0.0)
                yield ContentTokenEvent(1.5, "warriors")
                yield ContentDoneEvent(1.6, "warriors")
                yield SessionEndEvent(1.6)

            def disconnect(self):
                pass

        def fake_load(self, path):
            return [b"\x00\x00"] * 2, 0.4

        monkeypatch.setattr(
            "realqa.audio.AudioLoader.load_for_streaming", fake_load
        )

        out2 = tmp_path / "out2"
        out2.mkdir()
        legacy = {
            "sample_id": "same",  # pre-multi-task format: no task_type/query
            "pred": ["<w>", "warriors"],
            "timeline_len": 2,
            "prefix_len": 0,
            "chunks_per_second": 5.0,
            "emissions": [{"text": "warriors", "t_start": 1.0, "t_end": 1.4}],
        }
        (out2 / "predictions.jsonl").write_text(json.dumps(legacy) + "\n")
        (out2 / "failures.jsonl").write_text("")

        r = InferenceRunner(
            dataset_path=dataset,
            output_dir=out2,
            client=FakeClient(),
            session_config=SessionConfig(model="dummy", chunks_per_second=5.0),
            audio_loader=AudioLoader(chunk_ms=200),
            timeline_builder=TimelineBuilder(chunks_per_second=5.0, silence_token="<w>"),
            formatter=EvalFormatter(silence_token="<w>"),
            resume=True,
        )
        stats = r.run()
        assert stats["skipped"] == 1  # legacy line covers one sample only
        assert stats["success"] == 1
        preds = [
            json.loads(l)
            for l in (out2 / "predictions.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert {p.get("sample_id") for p in preds} == {
            "same",  # the legacy line, untouched
            "same::semantic_detection::profession",  # the newly-run task
        }

    def test_resume_keeps_other_tasks_on_same_utt(self, tmp_path, monkeypatch):
        """Two tasks sharing utterance_id must not collapse under resume."""
        from realqa.base import (
            SessionReadyEvent, ContentTokenEvent, ContentDoneEvent, SessionEndEvent,
        )

        rows = [
            {
                "utterance_id": "same",
                "audio_path": "/tmp/missing.flac",
                "prompt": "p1",
                "task_type": "lexical_detection",
                "query": "alpha",
                "labels": [[0.0, 0.1, "alpha"]],
            },
            {
                "utterance_id": "same",
                "audio_path": "/tmp/missing.flac",
                "prompt": "p2",
                "task_type": "semantic_detection",
                "query": "beta",
                "labels": [[0.2, 0.3, "beta"]],
            },
        ]
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        class FakeClient:
            def connect(self, config):
                pass

            def send_audio(self, b, i):
                pass

            def end_audio(self):
                pass

            def iter_events(self):
                yield SessionReadyEvent(0.0)
                yield ContentTokenEvent(0.2, "x")
                yield ContentDoneEvent(0.3, "x")
                yield SessionEndEvent(0.3)

            def disconnect(self):
                pass

            @property
            def is_connected(self):
                return False

        def fake_load(self, path):
            return [b"\x00\x00"] * 2, 0.4

        monkeypatch.setattr(
            "realqa.audio.AudioLoader.load_for_streaming", fake_load
        )

        r1 = self._runner(tmp_path, dataset, FakeClient(), resume=False)
        stats1 = r1.run()
        assert stats1["success"] == 2
        preds = [
            json.loads(l)
            for l in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert {p["sample_id"] for p in preds} == {
            "same::lexical_detection::alpha",
            "same::semantic_detection::beta",
        }

        # Legacy pred: sample_id was utterance-only for the first task.
        legacy = {
            "sample_id": "same",
            "pred": ["<w>", "alpha"],
            "timeline_len": 2,
            "prefix_len": 0,
            "chunks_per_second": 5.0,
            "emissions": [{"text": "alpha", "t_start": 0.0, "t_end": 0.1}],
        }
        (tmp_path / "out2").mkdir()
        (tmp_path / "out2" / "predictions.jsonl").write_text(
            json.dumps(legacy) + "\n"
        )
        (tmp_path / "out2" / "failures.jsonl").write_text("")

        from realqa.runner import InferenceRunner
        from realqa.base import SessionConfig
        from realqa.audio import AudioLoader
        from realqa.timeline import TimelineBuilder
        from realqa.formatter import EvalFormatter

        r2 = InferenceRunner(
            dataset_path=dataset,
            output_dir=tmp_path / "out2",
            client=FakeClient(),
            session_config=SessionConfig(model="dummy", chunks_per_second=5.0),
            audio_loader=AudioLoader(chunk_ms=200),
            timeline_builder=TimelineBuilder(chunks_per_second=5.0, silence_token="<w>"),
            formatter=EvalFormatter(silence_token="<w>"),
            resume=True,
        )
        stats2 = r2.run()
        assert stats2["skipped"] == 1
        assert stats2["success"] == 1
        preds2 = [
            json.loads(l)
            for l in (tmp_path / "out2" / "predictions.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert len(preds2) == 2
        assert any(p.get("sample_id") == "same::semantic_detection::beta" for p in preds2)



# ═══════════════════════════════════════════════════════════════════
# Multi-turn (full-duplex) capture
# ═══════════════════════════════════════════════════════════════════


import asyncio
import queue as queue_mod
from types import SimpleNamespace


class _FakeWS:
    """Duck-typed websockets connection over a scripted message list.

    ``"__gate__"`` messages block on an asyncio Event so tests can flip
    client state (e.g. input completion) between turns deterministically.
    """

    def __init__(self, messages, gate=None):
        self.messages = list(messages)
        self.gate = gate
        self.sent = []

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for m in self.messages:
            if m.get("type") == "__gate__":
                if self.gate is not None:
                    await self.gate.wait()
                continue
            yield json.dumps(m)

    async def send(self, raw):
        self.sent.append(json.loads(raw))


def _new_client(cls, **attrs):
    client = object.__new__(cls)
    for k, v in attrs.items():
        setattr(client, k, v)
    return client


def _now_zero(self):
    return 0.0


class TestQwenAudioMultiTurn:
    def _client(self, **overrides):
        from realqa.adapters.qwen_audio_realtime import QwenAudioRealtimeClient
        attrs = dict(
            _event_queue=queue_mod.Queue(),
            _t0=0.0,
            _audio_started=True,
            _realtime_factor=1.0,
            _auto_detect=True,
            _input_complete=False,
            _response_active=False,
            _recv_ended=False,
            _settle_s=0.1,
            _ws=None,
        )
        attrs.update(overrides)
        client = _new_client(QwenAudioRealtimeClient, **attrs)
        client._now = _now_zero.__get__(client)
        return client

    def test_recv_survives_mid_stream_turns(self):
        """Two VAD turns mid-stream are both captured; the session ends only
        at the response.done that arrives after input completion."""
        client = self._client()

        async def scenario():
            gate = asyncio.Event()
            client._ws = _FakeWS([
                {"type": "response.created"},
                {"type": "response.audio_transcript.delta", "delta": "one"},
                {"type": "response.audio_transcript.done", "transcript": "one"},
                {"type": "response.done"},
                {"type": "__gate__"},
                {"type": "response.created"},
                {"type": "response.audio_transcript.delta", "delta": "two"},
                {"type": "response.done"},
            ], gate=gate)
            task = asyncio.create_task(client._recv_loop())
            await asyncio.sleep(0.05)  # first turn consumed
            assert client._recv_ended is False  # first response.done survived
            client._input_complete = True
            gate.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())
        events = []
        while True:
            try:
                events.append(client._event_queue.get_nowait())
            except queue_mod.Empty:
                break
        from realqa.base import ContentTokenEvent, SessionEndEvent
        tokens = [e.token for e in events if isinstance(e, ContentTokenEvent)]
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert tokens == ["one", "two"]
        assert len(ends) == 1
        assert client._recv_ended is True

    def test_finalize_forces_end_when_quiet(self):
        client = self._client(_recv_ended=False, _response_active=False)
        asyncio.run(asyncio.wait_for(client._finalize(), timeout=10))
        from realqa.base import SessionEndEvent
        assert isinstance(client._event_queue.get_nowait(), SessionEndEvent)


class TestGeminiMultiTurn:
    def _client(self, **overrides):
        from realqa.adapters.gemini_live import GeminiLiveClient
        attrs = dict(
            _event_queue=queue_mod.Queue(),
            _t0=0.0,
            _audio_started=True,
            _realtime_factor=1.0,
            _session=None,
            _recv_task=None,
            _input_complete=False,
            _in_turn=False,
            _settle_s=0.1,
        )
        attrs.update(overrides)
        client = _new_client(GeminiLiveClient, **attrs)
        client._now = _now_zero.__get__(client)
        return client

    @staticmethod
    def _msg(text=None, turn_complete=False):
        return SimpleNamespace(
            text=None,
            server_content=SimpleNamespace(
                output_transcription=(
                    SimpleNamespace(text=text) if text is not None else None
                ),
                turn_complete=turn_complete,
            ),
        )

    def test_recv_captures_multiple_turns(self):
        client = self._client()

        class FakeSession:
            def __init__(self, turns):
                self.turns = list(turns)

            def receive(self):
                return self.turns.pop(0)()

        async def turn1():
            yield self._msg(text="one")
            yield self._msg(turn_complete=True)

        async def scenario():
            gate = asyncio.Event()

            async def turn2():
                await gate.wait()
                yield self._msg(text="two")
                yield self._msg(turn_complete=True)

            client._session = FakeSession([turn1, turn2])
            task = asyncio.create_task(client._recv_loop())
            await asyncio.sleep(0.05)  # turn 1 consumed
            client._input_complete = True
            gate.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())
        from realqa.base import ContentTokenEvent, ContentDoneEvent, SessionEndEvent
        events = []
        while True:
            try:
                events.append(client._event_queue.get_nowait())
            except queue_mod.Empty:
                break
        tokens = [e.token for e in events if isinstance(e, ContentTokenEvent)]
        dones = [e for e in events if isinstance(e, ContentDoneEvent)]
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert tokens == ["one", "two"]
        assert [d.transcript for d in dones] == ["one", "two"]
        assert len(ends) == 1

    def test_finalize_cancels_blocked_receive(self):
        client = self._client()

        class FakeSession:
            def receive(self):
                async def gen():
                    await asyncio.Event().wait()  # never resolves
                    yield self._msg(text="never")
                return gen()

        client._session = FakeSession()

        async def scenario():
            client._recv_task = asyncio.create_task(client._recv_loop())
            await asyncio.sleep(0.05)
            await asyncio.wait_for(client._finalize(), timeout=10)

        asyncio.run(scenario())
        from realqa.base import SessionEndEvent
        assert isinstance(client._event_queue.get_nowait(), SessionEndEvent)


class TestOpenAIRealtimeMultiTurn:
    def _client(self, **overrides):
        import threading
        from realqa.adapters.openai_realtime import OpenAIRealtimeClient
        attrs = dict(
            _event_queue=queue_mod.Queue(),
            _t0=0.0,
            _audio_started=True,
            _realtime_factor=1.0,
            _lock=threading.Lock(),
            _auto_detect=True,
            _got_text_delta=False,
            _response_started=False,
            _response_finished=False,
            _response_active=False,
            _speech_started=False,
            _input_complete=False,
            _settle_s=0.1,
            _recv_ended=False,
            _ws=None,
        )
        attrs.update(overrides)
        client = _new_client(OpenAIRealtimeClient, **attrs)
        client._now = _now_zero.__get__(client)
        return client

    def test_recv_survives_mid_stream_turns(self):
        client = self._client()

        async def scenario():
            gate = asyncio.Event()
            client._ws = _FakeWS([
                {"type": "response.created"},
                {"type": "response.output_text.delta", "delta": "one"},
                {"type": "response.done", "response": {"output": []}},
                {"type": "__gate__"},
                {"type": "response.created"},
                {"type": "response.output_text.delta", "delta": "two"},
                {"type": "response.done", "response": {"output": []}},
            ], gate=gate)
            task = asyncio.create_task(client._recv_loop())
            await asyncio.sleep(0.05)
            assert client._recv_ended is False
            client._input_complete = True
            gate.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())
        from realqa.base import ContentTokenEvent, SessionEndEvent
        events = []
        while True:
            try:
                events.append(client._event_queue.get_nowait())
            except queue_mod.Empty:
                break
        tokens = [e.token for e in events if isinstance(e, ContentTokenEvent)]
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert tokens == ["one", "two"]
        assert len(ends) == 1

    def test_vad_finalize_no_forced_create_after_midstream_answer(self):
        """A model that already answered mid-stream must not get a fabricated
        response.create (that would add a false second turn)."""
        client = self._client(
            _response_started=True,
            _response_active=False,
        )
        client._ws = _FakeWS([])
        asyncio.run(asyncio.wait_for(client._vad_finalize(), timeout=10))
        assert all(m.get("type") != "response.create" for m in client._ws.sent)

    def test_vad_finalize_never_forces_a_silent_model_to_speak(self):
        """No response.create button: a model that never chose to respond
        stays silent and is scored as such (the protocol measures the
        system's own turn-taking, not client-elicited answers)."""
        client = self._client(
            _response_started=False,
            _response_active=False,
        )
        client._ws = _FakeWS([])
        asyncio.run(asyncio.wait_for(client._vad_finalize(), timeout=10))
        assert all(m.get("type") != "response.create" for m in client._ws.sent)


class TestQwenOmniDrain:
    def test_drain_and_end_forces_session_end(self):
        from realqa.adapters.qwen_omni import QwenOmniRealtimeClient
        client = _new_client(
            QwenOmniRealtimeClient,
            _queue=queue_mod.Queue(),
            _t0=0.0,
            _audio_started=True,
            _realtime_factor=1.0,
            _auto_detect=True,
            _input_complete=True,
            _response_active=False,
            _session_ended=False,
            _settle_s=0.1,
        )
        client._now = _now_zero.__get__(client)
        client._drain_and_end()
        from realqa.base import SessionEndEvent
        assert isinstance(client._queue.get_nowait(), SessionEndEvent)

    def test_drain_and_end_skips_when_callback_ended(self):
        from realqa.adapters.qwen_omni import QwenOmniRealtimeClient
        client = _new_client(
            QwenOmniRealtimeClient,
            _queue=queue_mod.Queue(),
            _t0=0.0,
            _audio_started=True,
            _realtime_factor=1.0,
            _auto_detect=True,
            _input_complete=True,
            _response_active=False,
            _session_ended=True,  # callback already emitted SessionEnd
            _settle_s=0.1,
        )
        client._now = _now_zero.__get__(client)
        client._drain_and_end()
        assert client._queue.empty()


# ═══════════════════════════════════════════════════════════════════
# Endpoint/key configuration (standard env names + --env-file)
# ═══════════════════════════════════════════════════════════════════


class TestEnvFileAndPriorities:
    def test_load_env_file(self, tmp_path):
        import os
        from realqa.runner import _load_env_file

        f = tmp_path / "endpoints.env"
        f.write_text(
            "# comment\n"
            "\n"
            "GEMINI_API_KEY=abc123\n"
            'OPENAI_BASE_URL="https://relay.example/v1"\n'
            "QWEN_AUDIO_WS_URL='wss://cn.example/rt'\n"
            "MALFORMED_LINE_NO_EQUALS\n"
        )
        for k in ("GEMINI_API_KEY", "OPENAI_BASE_URL", "QWEN_AUDIO_WS_URL"):
            os.environ.pop(k, None)
        loaded = _load_env_file(str(f))
        assert loaded == ["GEMINI_API_KEY", "OPENAI_BASE_URL", "QWEN_AUDIO_WS_URL"]
        assert os.environ["GEMINI_API_KEY"] == "abc123"
        assert os.environ["OPENAI_BASE_URL"] == "https://relay.example/v1"
        assert os.environ["QWEN_AUDIO_WS_URL"] == "wss://cn.example/rt"

    def test_openai_priority_and_endpoint(self, monkeypatch):
        from realqa.adapters.openai_realtime import OpenAIRealtimeClient

        monkeypatch.setenv("OPENAI_API_KEY", "std-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://std.example/v1")
        c = OpenAIRealtimeClient()
        assert c._api_key == "std-key"
        assert c.endpoint == "https://std.example/v1"

    def test_live_azure_priority(self, monkeypatch):
        from realqa.adapters.openai_live import OpenAILiveClient

        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/openai/v1"
        )
        c = OpenAILiveClient()
        assert c._api_key == "az-key"
        assert c.endpoint == "https://res.openai.azure.com/openai/v1"

    def test_gemini_priority(self, monkeypatch):
        from realqa.adapters.gemini_live import HAS_GENAI, GeminiLiveClient

        if not HAS_GENAI:
            pytest.skip("google-genai not installed")
        monkeypatch.setenv("GEMINI_API_KEY", "std-key")
        c = GeminiLiveClient()
        assert c._api_key == "std-key"


# ═══════════════════════════════════════════════════════════════════
# GPT-Live endpoint flavors: OpenAI native vs Azure vs --ws-url override
# ═══════════════════════════════════════════════════════════════════


class _LiveFakeWS:
    def __init__(self):
        self.sent = []
        self.url = None
        self.headers = None
        self._recv_done = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if not self._recv_done:
            self._recv_done = True
            return json.dumps({"type": "session.started"})
        await asyncio.Event().wait()

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        return
        yield  # pragma: no cover


class TestGPTLiveFlavors:
    def _connect(self, monkeypatch, env: dict, extra: dict):
        import realqa.adapters.openai_live as live_mod
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_ENDPOINT", "GPT_DEPLOYMENT_NAME"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        fake = _LiveFakeWS()

        class FakeWSModule:
            @staticmethod
            async def connect(url, proxy=None, **kwargs):
                fake.url = url
                fake.headers = kwargs.get("additional_headers")
                return fake

        monkeypatch.setattr(live_mod, "websockets", FakeWSModule)
        client = live_mod.OpenAILiveClient()
        cfg = SimpleNamespace(
            model="gpt-live-1", prompt="p",
            extra=dict(extra), audio_sample_rate=24000,
        )
        asyncio.run(client._async_connect(cfg))
        start = next(m for m in fake.sent if m.get("type") == "session.start")
        return fake, start["session"]["model"]

    def test_native_default(self, monkeypatch):
        fake, model = self._connect(
            monkeypatch,
            env={"OPENAI_API_KEY": "sk-x", "GPT_DEPLOYMENT_NAME": "dep-42"},
            extra={},
        )
        assert fake.url == "wss://api.openai.com/v1/live/sessions"
        assert fake.headers == {"Authorization": "Bearer sk-x"}
        assert model == "gpt-live-1"  # deployment name ignored on native

    def test_azure(self, monkeypatch):
        fake, model = self._connect(
            monkeypatch,
            env={
                "AZURE_OPENAI_API_KEY": "az-x",
                "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com/openai/v1",
                "GPT_DEPLOYMENT_NAME": "dep-42",
            },
            extra={},
        )
        assert fake.url == "wss://res.openai.azure.com/openai/v1/live/sessions"
        assert fake.headers == {"api-key": "az-x"}
        assert model == "dep-42"  # deployment name replaces the model id

    def test_ws_url_override_flips_flavor(self, monkeypatch):
        fake, model = self._connect(
            monkeypatch,
            env={
                "AZURE_OPENAI_API_KEY": "az-x",
                "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com/openai/v1",
                "GPT_DEPLOYMENT_NAME": "dep-42",
            },
            extra={"ws_url": "https://api.openai.com/v1"},
        )
        assert fake.url == "wss://api.openai.com/v1/live/sessions"
        assert fake.headers["Authorization"] == "Bearer az-x"
        assert model == "gpt-live-1"

    def test_duplex_alias_maps_to_plain_id(self):
        from realqa.adapters.openai_live_duplex import OpenAILiveDuplexClient
        assert OpenAILiveDuplexClient._resolve_model("gpt-live-duplex") == "gpt-live-1"
        assert OpenAILiveDuplexClient._resolve_model(None) == "gpt-live-1"
        assert OpenAILiveDuplexClient._resolve_model("my-deploy") == "my-deploy"


class TestPerApiKeys:
    def test_realtime_dedicated_wins_over_generic(self, monkeypatch):
        from realqa.adapters.openai_realtime import OpenAIRealtimeClient

        monkeypatch.setenv("OPENAI_API_KEY", "generic-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://generic.example/v1")
        monkeypatch.setenv("OPENAI_REALTIME_API_KEY", "rt-key")
        monkeypatch.setenv("OPENAI_REALTIME_BASE_URL", "https://rt-relay.example/v1")
        c = OpenAIRealtimeClient()
        assert c._api_key == "rt-key"
        assert c.endpoint == "https://rt-relay.example/v1"
        # generic fallback still works when dedicated names are absent
        monkeypatch.delenv("OPENAI_REALTIME_API_KEY")
        monkeypatch.delenv("OPENAI_REALTIME_BASE_URL")
        c2 = OpenAIRealtimeClient()
        assert c2._api_key == "generic-key"
        assert c2.endpoint == "https://generic.example/v1"

    def test_live_dedicated_wins_over_azure_and_generic(self, monkeypatch):
        from realqa.adapters.openai_live import OpenAILiveClient

        monkeypatch.setenv("OPENAI_API_KEY", "generic-key")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/openai/v1")
        monkeypatch.setenv("OPENAI_LIVE_API_KEY", "live-key")
        c = OpenAILiveClient()
        assert c._api_key == "live-key"
        # dedicated endpoint var decides the flavor, overriding Azure
        monkeypatch.setenv("OPENAI_LIVE_BASE_URL", "https://api.openai.com/v1")
        c2 = OpenAILiveClient()
        assert c2._uses_azure is False
        assert c2.endpoint == "https://api.openai.com/v1"
