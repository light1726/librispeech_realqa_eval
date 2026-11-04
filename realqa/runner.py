"""Inference runner + CLI.

Orchestrates the full pipeline:
  Dataset → API Client → Timeline → Formatter → Eval JSONL

Run as::

    # Separate runs per LibriSpeech split (recommended):
    python -m realqa.runner \\
        --split test-clean \\
        --model qwen-audio-3.0-realtime-flash \\
        --turn-detection smart_turn

    python -m realqa.runner \\
        --split test-other \\
        --model qwen-audio-3.0-realtime-flash \\
        --turn-detection smart_turn \\
        --resume

    # Or pass dataset/output explicitly:
    python -m realqa.runner \\
        --dataset /path/to/test-clean.jsonl \\
        --output-dir output/qwen_audio_flash_test-clean \\
        --model qwen-audio-3.0-realtime-flash

    # Private / regional endpoint (URL + key via CLI/env, not code defaults):
    python -m realqa.runner \\
        --split test-clean \\
        --model qwen-audio-3.0-realtime-flash \\
        --ws-url wss://…/api-ws/v1/realtime \\
        --api-key-env PROVIDER_API_KEY \\
        --turn-detection server_vad
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from realqa.dataset import DatasetReader
from realqa.audio import AudioLoader
from realqa.base import (
    ErrorEvent,
    SessionConfig,
    SessionEndEvent,
    StreamEvent,
)
from realqa.formatter import EvalFormatter
from realqa.timeline import TimelineBuilder
from realqa.sample import (
    DatasetSample,
    emissions_fingerprint,
    make_sample_id,
)


class InferenceRunner:
    """Orchestrates one evaluation run over a dataset.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the benchmark JSONL file.
    output_dir : str or Path
        Directory for ``predictions.jsonl``, ``failures.jsonl``, and
        ``run_stats.json``.
    client : StreamingClient
        Instantiated API client (e.g. ``QwenOmniRealtimeClient``).
    session_config : SessionConfig
        Model-agnostic session parameters.
    audio_loader : AudioLoader
        Audio preprocessing tool.
    timeline_builder : TimelineBuilder
        Event-to-timeline converter.
    formatter : EvalFormatter
        Sample + timeline → JSONL line formatter.
    task_type : str or None
        Optional task-type filter.
    sample_limit : int or None
        Maximum number of samples to process.
    seed : int or None
        Shuffle seed before limiting.
    timeout_s : float
        Per-sample timeout in seconds.
    resume : bool
        If True, append to existing outputs and skip sample_ids already
        present in ``predictions.jsonl``.  Failed ids in ``failures.jsonl``
        are retried; on success they are removed from failures.

        Sample identity is ``utterance_id::task_type::query``.  Legacy
        prediction lines that only stored ``utterance_id`` (no task/query)
        are matched via GT emission fingerprints; each legacy line covers
        exactly one pending sample (consume-on-match).  Fresh (non-resume)
        runs never skip: every dataset row is processed.
    """

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        output_dir: str | Path,
        client,
        session_config: SessionConfig,
        audio_loader: AudioLoader,
        timeline_builder: TimelineBuilder,
        formatter: EvalFormatter,
        task_type: Optional[str] = None,
        sample_limit: Optional[int] = None,
        seed: Optional[int] = None,
        timeout_s: float = 120.0,
        resume: bool = False,
        realtime_factor: float = 1.0,
        run_config: Optional[dict] = None,
    ):
        self._dataset_path = Path(dataset_path)
        self._output_dir = Path(output_dir)
        self._client = client
        self._session_config = session_config
        self._audio_loader = audio_loader
        self._timeline_builder = timeline_builder
        self._formatter = formatter
        self._task_type = task_type
        self._sample_limit = sample_limit
        self._seed = seed
        self._timeout_s = timeout_s
        self._resume = resume
        # >1 sends audio faster than realtime; 0 = as-fast-as-possible.
        self._realtime_factor = float(realtime_factor)
        # Non-secret run provenance merged into run_stats.json.
        self._run_config = dict(run_config or {})

    @staticmethod
    def _row_sample_id(row: dict) -> Optional[str]:
        """Resolve a composite sample_id from a predictions/failures JSONL row."""
        sid = row.get("sample_id")
        task = row.get("task_type")
        query = row.get("query")
        utt = row.get("utterance_id")
        if isinstance(sid, str) and "::" in sid:
            return sid
        if task is not None and query is not None:
            utt = utt or (sid if isinstance(sid, str) else None)
            if utt:
                return make_sample_id(str(utt), str(task), str(query))
        return None

    @classmethod
    def _load_completed(cls, path: Path) -> tuple[set, Counter]:
        """Load completed composite sample ids + legacy emission fingerprints.

        Fingerprints are harvested only from legacy rows that carry no
        task_type/query (no composite id exists there); each such row
        covers exactly one pending sample.  New-format rows contribute
        their ``sample_id`` only — identical labels across different
        tasks must not collide.
        """
        done_ids: set = set()
        legacy_fps: Counter = Counter()
        if not path.is_file():
            return done_ids, legacy_fps
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = cls._row_sample_id(row)
                if sid is not None:
                    done_ids.add(sid)
                    continue
                utt = row.get("utterance_id")
                if utt is None and isinstance(row.get("sample_id"), str):
                    # Legacy: sample_id was the utterance id alone.
                    utt = row["sample_id"].split("::", 1)[0]
                emissions = row.get("emissions") or []
                if utt is not None and emissions:
                    legacy_fps[emissions_fingerprint(str(utt), emissions)] += 1
        return done_ids, legacy_fps

    @classmethod
    def _load_failed_ids(cls, path: Path) -> set:
        """Load composite sample ids from failures.jsonl."""
        ids: set = set()
        if not path.is_file():
            return ids
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = cls._row_sample_id(row)
                if sid is not None:
                    ids.add(sid)
        return ids

    def _rewrite_failures_excluding(self, failures_path: Path, drop_ids: set) -> None:
        """Rewrite failures.jsonl without *drop_ids*, keeping the latest row per id."""
        if not failures_path.is_file():
            return
        by_id: Dict[str, dict] = {}
        orphans: List[str] = []
        with failures_path.open() as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    orphans.append(raw + "\n")
                    continue
                sid = self._row_sample_id(row)
                if sid is None or sid in drop_ids:
                    continue
                by_id[sid] = row
        with failures_path.open("w") as fh:
            for row in by_id.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.writelines(orphans)

    def run(self) -> Dict[str, int]:
        """Execute the full pipeline.

        Returns
        -------
        dict
            Counts: ``{"total": N, "success": N, "error": N, "skipped": N}``.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_out = self._output_dir / "predictions.jsonl"
        failures_out = self._output_dir / "failures.jsonl"
        stats_out = self._output_dir / "run_stats.json"

        resume_ids: set = set()
        legacy_fps: Counter = Counter()
        prior_failed: set = set()
        if self._resume:
            resume_ids, legacy_fps = self._load_completed(jsonl_out)
            prior_failed = self._load_failed_ids(failures_out)
            pred_mode, fail_mode = "a", "a"
            print(
                f"Resume: skipping {len(resume_ids)} completed id(s) "
                f"(+{sum(legacy_fps.values())} legacy fingerprint match(es)); "
                f"{len(prior_failed)} prior failure(s) may be retried",
                flush=True,
            )
        else:
            pred_mode, fail_mode = "w", "w"

        reader = DatasetReader(
            self._dataset_path,
            task_type=self._task_type,
            limit=self._sample_limit,
            seed=self._seed,
        )

        total, success, errors, skipped = 0, 0, 0, 0
        recovered: set = set()  # previously failed, now succeeded
        seen_ids: set = set()  # dataset duplicate guard (warn, never skip)

        with jsonl_out.open(pred_mode) as pred_fh, failures_out.open(fail_mode) as fail_fh:
            for sample in reader:
                total += 1
                sid = sample.sample_id
                if sid in seen_ids:
                    print(
                        f"  WARNING: duplicate sample_id {sid!r} in dataset; "
                        "running it again",
                        flush=True,
                    )
                seen_ids.add(sid)
                if sid in resume_ids:
                    skipped += 1
                    print(f"  [{total}] {sid}  SKIP (already done)", flush=True)
                    continue
                fp = sample.fingerprint
                if fp in legacy_fps:
                    # One legacy prediction line covers exactly one sample.
                    legacy_fps[fp] -= 1
                    if legacy_fps[fp] <= 0:
                        del legacy_fps[fp]
                    skipped += 1
                    print(
                        f"  [{total}] {sid}  SKIP (legacy fingerprint match)",
                        flush=True,
                    )
                    continue
                try:
                    line = self._process_sample(sample)
                    pred_fh.write(line)
                    pred_fh.flush()
                    success += 1
                    if sid in prior_failed:
                        recovered.add(sid)
                except Exception as exc:
                    errors += 1
                    err_msg = str(exc)
                    print(f"  ERROR [{sid}]: {err_msg}", file=sys.stderr, flush=True)
                    fail_fh.write(json.dumps({
                        "sample_id": sid,
                        "utterance_id": sample.utterance_id,
                        "task_type": sample.task_type,
                        "query": sample.query,
                        "audio_path": sample.audio_path,
                        "error": err_msg,
                        "timestamp": time.time(),
                    }, ensure_ascii=False) + "\n")
                    fail_fh.flush()
                print(f"  [{total}] {sid}  "
                      f"task={sample.task_type}  query={sample.query}",
                      flush=True)

        # Drop recovered ids and dedupe failures (keep latest per utterance).
        self._rewrite_failures_excluding(failures_out, recovered)

        stats = {
            "total": total,
            "success": success,
            "error": errors,
            "skipped": skipped,
            "resume": self._resume,
            "predictions_path": str(jsonl_out),
            "failures_path": str(failures_out),
            "config": self._run_config,
        }
        with stats_out.open("w") as sfh:
            json.dump(stats, sfh, indent=2)
        return stats

    def _process_sample(self, sample: DatasetSample) -> str:
        """Run one sample through the API → timeline → format pipeline."""
        chunks, audio_duration = self._audio_loader.load_for_streaming(
            sample.audio_path
        )

        # Configure session for this sample's prompt
        config = SessionConfig(
            model=self._session_config.model,
            prompt=sample.prompt,
            chunks_per_second=self._session_config.chunks_per_second,
            audio_sample_rate=self._session_config.audio_sample_rate,
            audio_encoding=self._session_config.audio_encoding,
            extra=dict(self._session_config.extra),
        )
        config.extra["realtime_factor"] = self._realtime_factor

        self._client.connect(config)
        try:
            chunk_s = self._audio_loader.chunk_ms / 1000.0
            factor = self._realtime_factor
            sleep_s = 0.0 if factor <= 0 else (chunk_s / factor)
            for i, chunk in enumerate(chunks):
                self._client.send_audio(chunk, i)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            # Silence padding for VAD / GPT-Live turn-taking (end-of-speech cue).
            silence_sec = config.extra.get("silence_padding_s")
            if silence_sec is None and config.extra.get("turn_detection") is not None:
                silence_sec = 1.5
            if silence_sec:
                silence_chunk = b"\x00" * self._audio_loader.chunk_bytes
                n_silence = int(float(silence_sec) / chunk_s)
                for j in range(n_silence):
                    self._client.send_audio(silence_chunk, len(chunks) + j)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            self._client.end_audio()

            # Prefer client-side idle timeout when supported (OpenAI VAD can hang
            # if the server never emits response.done).
            try:
                events = list(self._client.iter_events(timeout_s=self._timeout_s))
            except TypeError:
                events = list(self._client.iter_events())
        finally:
            self._client.disconnect()

        # Check for errors
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        if error_events:
            raise RuntimeError(
                f"API errors: {[e.message for e in error_events]}"
            )

        timeline = self._timeline_builder.build(
            events,
            audio_duration,
            prefix_len=self._session_config.extra.get("prefix_len", 0),
        )

        return self._formatter.format_line_json(sample, timeline)


# ── CLI ────────────────────────────────────────────────────────


def _get_client(
    model: str,
    *,
    voice: str = "Ethan",
    api_key: Optional[str] = None,
):
    """Resolve a model name to a concrete client instance."""
    m = model.lower()
    if "gemini" in m:
        from realqa.adapters.gemini_live import GeminiLiveClient
        return GeminiLiveClient(api_key=api_key)
    if "qwen-audio" in m:
        from realqa.adapters.qwen_audio_realtime import QwenAudioRealtimeClient
        return QwenAudioRealtimeClient(api_key=api_key)
    if "qwen" in m:
        from realqa.adapters.qwen_omni import (
            QwenOmniRealtimeClient,
        )
        return QwenOmniRealtimeClient(api_key=api_key)
    if m.startswith("gpt-live") and "duplex" in m:
        from realqa.adapters.openai_live_duplex import OpenAILiveDuplexClient
        return OpenAILiveDuplexClient(api_key=api_key, voice=voice)
    if m.startswith("gpt-live"):
        from realqa.adapters.openai_live import OpenAILiveClient
        return OpenAILiveClient(api_key=api_key, voice=voice)
    if "gpt-realtime" in m:
        from realqa.adapters.openai_realtime import OpenAIRealtimeClient
        return OpenAIRealtimeClient(api_key=api_key)
    raise ValueError(
        f"Unknown model: {model!r}. "
        f"Supported: qwen3.5-omni-flash-realtime, qwen-audio-3.0-realtime-flash, "
        f"gemini-3.1-flash-live-preview, gpt-realtime-mini, gpt-live-1, gpt-live-duplex"
    )


# Default LibriSpeech realtime-QA benchmark root (separate JSONL per split).
DEFAULT_BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "data/benchmark"))
KNOWN_SPLITS = (
    "test-clean",
    "test-other",
    "dev-clean",
    "dev-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


def _model_slug(model: str) -> str:
    """Filesystem-friendly model id for default output dirs."""
    return (
        model.replace("models/", "")
        .replace(":", "-")
        .replace("/", "-")
        .replace(".", "-")
    )


def _resolve_dataset_and_output(args: argparse.Namespace) -> tuple[Path, Path, Optional[str]]:
    """Resolve dataset path, output dir, and split name from CLI args."""
    split = args.split
    if args.dataset:
        dataset = Path(args.dataset)
        if split is None:
            # Infer split from filename stem when possible.
            stem = dataset.stem.replace(".neg", "")
            if stem in KNOWN_SPLITS:
                split = stem
    elif split:
        root = Path(args.benchmark_root)
        dataset = root / f"{split}.jsonl"
    else:
        raise SystemExit("Provide --dataset and/or --split")

    if not dataset.is_file():
        raise SystemExit(f"Dataset not found: {dataset}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif split:
        output_dir = Path("output") / f"{_model_slug(args.model)}_{split}"
    else:
        raise SystemExit(
            "--output-dir is required when --split is not set "
            "(cannot derive a split-specific default)"
        )

    return dataset, output_dir, split


def _load_env_file(path: str) -> List[str]:
    """Load KEY=VALUE lines from a dotenv-style file into ``os.environ``.

    ``#`` comments and blank lines are skipped; surrounding quotes are
    stripped. The file only fills the same environment variables the
    adapters already read (standard provider names) — no other use.
    """
    loaded: List[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run streaming API inference and produce eval JSONL. "
                    "Use a separate --split / output dir for test-clean vs test-other.",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Path to benchmark JSONL. Optional if --split is set.",
    )
    p.add_argument(
        "--split", type=str, default=None, choices=KNOWN_SPLITS,
        help="Benchmark split (e.g. test-clean, test-other). "
             "Resolves dataset under --benchmark-root and defaults "
             "output-dir to output/<model>_<split>/.",
    )
    p.add_argument(
        "--benchmark-root", type=str, default=str(DEFAULT_BENCHMARK_ROOT),
        help="Directory with {split}.jsonl files "
             "(default: data/benchmark, override with BENCHMARK_ROOT).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for predictions.jsonl, failures.jsonl, run_stats.json. "
             "Default with --split: output/<model>_<split>/.",
    )
    p.add_argument("--model", type=str, default="qwen3.5-omni-flash-realtime",
                   help="Model identifier (qwen3.5-omni-flash-realtime, "
                        "qwen-audio-3.0-realtime-flash, gemini-3.1-flash-live-preview, "
                        "gpt-realtime-mini, gpt-live-1, gpt-live-duplex).")
    p.add_argument("--task", type=str, default=None,
                   help="Optional task_type filter (e.g. lexical_detection).")
    p.add_argument("--n", type=int, default=None,
                   help="Max number of samples to evaluate.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for sample ordering before limiting.")
    p.add_argument("--chunk-ms", type=int, default=200,
                   help="Streaming audio chunk duration in ms (default 200).")
    p.add_argument("--silence-token", type=str, default="<silence>",
                   help="Silence token string.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-sample timeout in seconds (default 120).")
    p.add_argument("--voice", type=str, default=None,
                   help="TTS voice (Qwen-Audio: longanqian, Omni: Ethan, "
                        "Gemini: Zephyr, GPT-Live: marin). Default depends on model.")
    p.add_argument("--turn-detection", type=str, default=None,
                   help="Turn detection mode: server_vad, semantic_vad, "
                   "smart_turn. Pass this for benchmark runs of OpenAI "
                   "Realtime / Qwen so the provider's turn-taking (not a "
                   "manual commit) elicits responses; without it those "
                   "backends are push-to-talk (connectivity debugging). "
                   "Gemini Live and GPT-Live decide on their own either way.")
    p.add_argument(
        "--vad-eagerness", type=str, default=None,
        choices=("low", "medium", "high", "auto"),
        help="OpenAI semantic_vad eagerness (low|medium|high|auto). "
             "Only applied when --turn-detection is set.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Continue a previous run: append outputs, skip samples already "
             "in predictions.jsonl, retry ids listed in failures.jsonl.",
    )
    p.add_argument(
        "--realtime-factor", type=float, default=1.0,
        help="Audio send speed vs realtime (1=realtime, 10=10x faster, "
             "0=as fast as possible). Event timestamps are scaled so hit/FA "
             "placement stays aligned to audio time, but response latency "
             "is real time and gets scaled with it: delay metrics are only "
             "valid at 1 (the default). Use other values for smoke tests "
             "only.",
    )
    p.add_argument(
        "--settle-s", type=float, default=2.0,
        help="VAD/duplex finalize settle window in seconds: after input "
             "ends and an in-flight response drains, how long to wait for "
             "one more late turn before forcing the session end "
             "(default: 2.0).",
    )
    p.add_argument(
        "--env-file", type=str, default=None,
        help="dotenv-style file with API keys/endpoints (KEY=VALUE lines; "
             "see endpoints.env.example). Loaded before clients are "
             "constructed; uses the standard provider variable names.",
    )
    p.add_argument(
        "--ws-url", type=str, default=None,
        help="WebSocket base URL for Qwen-Audio realtime (overrides "
             "QWEN_AUDIO_WS_URL / DASHSCOPE_WS_URL / adapter default). "
             "Example MaaS: wss://…maas.aliyuncs.com/api-ws/v1/realtime",
    )
    p.add_argument(
        "--api-key-env", type=str, default=None,
        help="Env var name holding the API key for private deployments. "
             "When omitted, adapters use their standard provider key env.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.env_file:
        if not os.path.isfile(args.env_file):
            raise SystemExit(f"--env-file not found: {args.env_file}")
        loaded = _load_env_file(args.env_file)
        print(f"env file {args.env_file}: loaded {len(loaded)} variable(s)")

    if args.chunk_ms <= 0:
        raise SystemExit("--chunk-ms must be positive")
    chunks_per_second = 1000.0 / args.chunk_ms

    dataset_path, output_dir, split = _resolve_dataset_and_output(args)

    model_l = args.model.lower()
    if args.voice is not None:
        voice = args.voice
    elif "gemini" in model_l:
        voice = "Zephyr"
    elif "qwen-audio" in model_l:
        voice = "longanqian"
    elif model_l.startswith("gpt-live"):
        voice = "marin"
    else:
        voice = "Ethan"

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(
                f"--api-key-env {args.api_key_env!r} is set but empty/unset"
            )

    client = _get_client(
        args.model,
        voice=voice,
        api_key=api_key,
    )
    extra: dict = {"voice": voice,
                   "settle_s": args.settle_s}
    if args.turn_detection:
        extra["turn_detection"] = {
            "type": args.turn_detection,
        }
        if args.vad_eagerness:
            extra["turn_detection"]["eagerness"] = args.vad_eagerness
        extra["silence_padding_s"] = 1.5
    elif args.vad_eagerness:
        raise SystemExit("--vad-eagerness requires --turn-detection")
    if args.ws_url:
        extra["ws_url"] = args.ws_url
    if "gemini" in model_l:
        # Native-audio Live default: AUDIO + output transcription.
        extra.setdefault("response_modality", "AUDIO")
    if model_l.startswith("gpt-live"):
        # Live is always duplex; silence cues end-of-utterance.
        # Duplex-tuned client: shorter pad so we lean less on post-EOA turns.
        if "duplex" in model_l:
            extra.setdefault("silence_padding_s", 0.4)
        else:
            extra.setdefault("silence_padding_s", 1.5)
    # OpenAI Realtime / Live default to 24 kHz PCM input.
    audio_sr = 24000 if ("gpt-realtime" in model_l or model_l.startswith("gpt-live")) else 16000
    session_config = SessionConfig(
        model=args.model,
        chunks_per_second=chunks_per_second,
        audio_sample_rate=audio_sr,
        extra=extra,
    )
    audio_loader = AudioLoader(target_sample_rate=audio_sr, chunk_ms=args.chunk_ms)
    timeline_builder = TimelineBuilder(
        chunks_per_second=chunks_per_second,
        silence_token=args.silence_token,
    )
    formatter = EvalFormatter(silence_token=args.silence_token)

    print(f"split={split or '(unspecified)'}  dataset={dataset_path}")
    print(f"output_dir={output_dir}")
    print(f"Streaming chunk_ms={args.chunk_ms}  chunks_per_second="
          f"{chunks_per_second:.4g}  realtime_factor={args.realtime_factor}")
    if args.realtime_factor != 1.0:
        print(
            "WARNING: realtime_factor != 1 — audio-time alignment holds for "
            "hit/FA placement, but delay/latency metrics from this run are "
            "INVALID (real response time is compressed by the factor). "
            "Smoke-test only; use the default 1.0 for evaluation runs.",
            flush=True,
        )

    runner = InferenceRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        client=client,
        session_config=session_config,
        audio_loader=audio_loader,
        timeline_builder=timeline_builder,
        formatter=formatter,
        task_type=args.task,
        sample_limit=args.n,
        seed=args.seed,
        timeout_s=args.timeout,
        resume=args.resume,
        realtime_factor=args.realtime_factor,
        run_config={
            "model": args.model,
            "provider": type(client).__name__,
            "endpoint": getattr(client, "endpoint", None),
            "env_file": args.env_file,
            "api_key_env": args.api_key_env,
            "ws_url": args.ws_url,
            "turn_detection": args.turn_detection,
            "vad_eagerness": args.vad_eagerness,
            "chunk_ms": args.chunk_ms,
            "chunks_per_second": chunks_per_second,
            "settle_s": args.settle_s,
            "realtime_factor": args.realtime_factor,
        },
    )

    stats = runner.run()
    print(f"Done.  total={stats['total']}  success={stats['success']}  "
          f"error={stats['error']}  skipped={stats['skipped']}")
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
