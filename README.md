# LibriSpeech Realtime QA — RealQA evaluation toolkit

End-to-end pipeline: stream LibriSpeech audio to realtime speech APIs,
record **when** and **what** each model answers, and score streaming vs
post-audio behavior against ground-truth emissions.

**[Interactive demo](https://zxzxz-1.github.io/librispeech-real-qa-demo/)** —
per-chunk model outputs on test-clean clips, with answer timing scored
against the queried word's onset.

```
benchmark JSONL ──▶ realqa.runner (API inference) ──▶ predictions.jsonl
                                                            │
                     realqa.eval ◀────────────────┘
                       (streaming / post-audio metrics)
```

## 1. Environment setup

### 1.1 Python dependencies

```bash
pip install -r requirements.txt
```

Installs the evaluation core (torch, numpy, soundfile, scipy) and the
backend SDKs (websockets for OpenAI Realtime / GPT-Live / Qwen-Audio,
google-genai for Gemini Live, dashscope for Qwen-Omni).

### 1.2 API keys and endpoints

Copy the template and fill in the providers you use:

```bash
cp endpoints.env.example endpoints.env   # gitignored; never commit it
```

Standard variable names (see the template for details):

| Provider | Variables |
|---|---|
| Gemini Live | ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) |
| GPT Realtime | ``OPENAI_REALTIME_API_KEY`` + ``OPENAI_REALTIME_BASE_URL`` (relay optional); falls back to ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL`` |
| GPT Live | ``OPENAI_LIVE_API_KEY`` + ``OPENAI_LIVE_BASE_URL``; or Azure: ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT`` + ``GPT_DEPLOYMENT_NAME`` |
| Qwen (DashScope) | ``DASHSCOPE_API_KEY`` (+ ``QWEN_AUDIO_WS_URL`` for the China endpoint) |

Pass the file to every runner invocation with ``--env-file endpoints.env``.
The resolved **non-secret** settings (model, endpoint, flags) are stamped
into each run's ``run_stats.json`` for provenance.

### 1.3 Network

All backends need egress to the provider. On restricted hosts set an HTTP
proxy before running (``export https_proxy=http://<proxy-host>:<port>``).

### 1.4 Benchmark data

One JSONL per split; each line is one (utterance × task × query) sample:

```json
{"utterance_id": "5142-33396-0002", "audio_path": "/…/5142-33396-0002.flac",
 "task_type": "lexical_detection", "query": "warriors",
 "prompt": "The moment you hear the word \"warriors\" …, say it out loud.",
 "labels": [[0.87, 1.31, "warriors"]]}
```

Point ``--dataset`` at a split file (``--benchmark-root`` + ``--split``
also work if split files live together).

## 2. Inference — run the APIs

**Response policy.** The evaluated system decides for itself when to
respond; earlier is better. The client never presses a push-to-talk
button: OpenAI Realtime and Qwen benchmark runs pass ``--turn-detection``
(the provider's turn-taking elicits responses), while Gemini Live and
GPT-Live respond on their own. A model that never speaks is scored as
such. Bare push-to-talk mode (no flag) is for connectivity debugging only.

**Timing.** Audio streams at true realtime by default. Keep
``--realtime-factor 1`` for any run you will report delays from —
accelerated runs compress real response latency and are for smoke tests.

All commands write to the output dir: ``predictions.jsonl`` (grid +
event-level ``pred_runs`` with exact answer timestamps + ground-truth
emissions), ``failures.jsonl`` (errored samples), ``run_stats.json``.
``--resume`` continues an interrupted run; ``--n K`` limits sample count.
Errored samples never reach evaluation — when publishing numbers, state
coverage (``n_samples`` scored vs ``run_stats.json`` totals).

**Gemini Live**

```bash
python -m realqa.runner --env-file endpoints.env \
  --dataset <benchmark>/test-clean.jsonl --output-dir out/gemini \
  --model gemini-3.1-flash-live-preview
```

**GPT Realtime** — use ``server_vad`` (through some relays
``semantic_vad`` never auto-responds and scores as silence):

```bash
python -m realqa.runner --env-file endpoints.env \
  --dataset <benchmark>/test-clean.jsonl --output-dir out/gptrealtime \
  --model gpt-realtime-mini --turn-detection server_vad
```

**GPT Live** — two access flavors; the effective endpoint decides
which: **OpenAI native** (the default — just ``OPENAI_API_KEY``, model
``gpt-live-1``) or **Azure** (``AZURE_OPENAI_API_KEY`` +
``AZURE_OPENAI_ENDPOINT`` + ``GPT_DEPLOYMENT_NAME``; the deployment name
replaces the model id, auth uses the Azure ``api-key`` header). A
``--ws-url https://api.openai.com/v1`` override flips an Azure-configured
machine to native for one run. Same model, two conditions: plain (raw
task prompt, 1.5 s silence pad) vs duplex (full-duplex instructions:
speak while the user talks, no backchannels, no preamble; 0.4 s pad,
faster finalize):

```bash
python -m realqa.runner --env-file endpoints.env \
  --dataset <benchmark>/test-clean.jsonl --output-dir out/gptlive \
  --model gpt-live-1          # or: --model gpt-live-duplex
```

**Qwen-Audio-3.0-Realtime** — with a China-mainland account, set
``QWEN_AUDIO_WS_URL`` in the env file and the command is uniform:

```bash
python -m realqa.runner --env-file endpoints.env \
  --dataset <benchmark>/test-clean.jsonl --output-dir out/qwen_audio \
  --model qwen-audio-3.0-realtime-flash --turn-detection server_vad
```

**Qwen3.5-Omni-Realtime** — same env file, uses the dashscope SDK:
``--model qwen3.5-omni-flash-realtime``.

## 3. Metrics calculation

Score a run's ``predictions.jsonl`` in either protocol (both read the
``pred_runs`` event timestamps automatically):

**Streaming** — measures realtime behavior: an answer counts as a hit
only if it starts inside the **acceptance window** ``[t_start, t_end + g]``
of the reference answer — never before the word starts, and at most ``g``
seconds after it ends. The onset is the arrival time of the answer's
first text delta at the client, on the audio-streaming clock (includes
provider network latency and delta batching — the same end-to-end rule
for every API). Three tiers are always reported:

| Tier | Window | Meaning |
|---|---|---|
| ``during_word`` | ``[t_start, t_end]`` | answered while the word was being spoken |
| ``+1c`` | ``[t_start, t_end + C]`` | answered within one chunk after the word ends |
| ``+2c`` | ``[t_start, t_end + 2C]`` | answered within two chunks after the word ends |

``C`` is the protocol chunk duration. For model timelines (rows carry
their own chunk rate) ``C = 1/chunks_per_second`` per row and no flag is
needed. ``--grace-window-sec`` is an
eval-side flag only (inference never uses it): when *evaluating* API
predictions, pass it to pin C, so the protocol does not depend on the
transport chunk size the rows happened to be collected with:

```bash
python -m realqa.eval out/<model>/predictions.jsonl \
  --output-dir out/<model>/eval --grace-window-sec 0.16
```

**Post-audio** — open-ended acceptance; measures eventual correctness and
response delay without the realtime constraint:

```bash
python -m realqa.eval out/gemini/predictions.jsonl \
  --output-dir out/gemini/eval_post --metric-mode post_audio
```

Outputs per evaluation: ``all_results.json`` (config + dataset-level +
per-sample results), ``summary.csv`` (one row of scalars), and per-sample
dump files. The CLI headline prints one line per grace window
(f1/precision/recall/fdr/delay from the same match set). Rows carry their
own ``chunks_per_second``/``prefix_len``; CLI flags are only defaults.

Evaluate task subsets separately (``task_type`` is not inferred):
filter the predictions file, or run one inference pass per task with
``--task``.

## Reference

### Prediction JSONL schema

```json
{
  "sample_id": "utt-001",
  "pred": ["<silence>", "...", "stew", "<|im_end|>", "..."],
  "pred_runs": [{"text": "stew", "t_start": 1.44, "t_end": 1.6}],
  "timeline_len": 116,
  "prefix_len": 33,
  "chunks_per_second": 6.25,
  "emissions": [{"text": "stew", "t_start": 1.44, "t_end": 1.83}]
}
```

- ``pred``: token-string list (preferred) or a string with the special
  tokens embedded — the **uniform interface** for all models and APIs;
  convert other formats (e.g. HF token ids) in preprocessing.
- ``pred_runs``: event-level predictions, one entry per model burst with
  the exact first-delta timestamp (written by ``realqa.runner``). When
  present, onsets are not frame-quantized and unit boundaries are the
  model's actual bursts rather than grid contiguity.
- ``timeline_len``: prediction grid length. ``chunks_per_second``: grid
  resolution — also the chunk rate C = 1/chunks_per_second that defines the
  acceptance windows. ``prefix_len``: ``1 + prompt_groups`` (nonzero only
  for silence-token model timelines; API timelines use 0).
  ``emissions``: ground-truth answers with ``text``/``t_start``/``t_end``
  in seconds.
- Legacy ``ref`` token grids are still accepted; ``emissions`` can be
  derived from them when absent.

### Eval CLI flags

| Flag | Purpose |
|------|---------|
| ``--silence-token`` | Silence / wait special token (segmentation) |
| ``--eos-token`` | Boundary token(s); skipped when decoding content |
| ``--chunks-per-second`` | Grid-resolution default; rows usually carry their own |
| ``--grace-window-sec`` | Evaluation-only flag: chunk duration C for ``during_word(0)/+1c(C)/+2c(2C)``. Omitted: C = 1/row's chunks_per_second (model timelines). Pass it when evaluating API predictions to pin the protocol independent of transport chunk size |
| ``--onset-source`` | Prediction units/timing: ``auto`` (events per row when ``pred_runs`` present, else grid), ``events`` (require ``pred_runs``), ``grid`` (always derive from the token grid; native causal timing for silence-token models) |
| ``--metric-mode`` | ``streaming`` (default) or ``post_audio`` |
| ``--early-eps-sec`` | post_audio only: reject answers starting > this many seconds before ``t_start`` (default 0.05) |
