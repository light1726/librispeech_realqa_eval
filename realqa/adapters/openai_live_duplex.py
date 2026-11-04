"""GPT-Live adapter tuned for full-duplex / mid-speech replies.

Keeps the same **per-utterance** session lifecycle as ``openai_live.py``
(connect → stream one clip → close).  Does **not** reuse sessions across
samples.

Changes vs ``OpenAILiveClient``
------------------------------
* Wraps the dataset prompt with Live interruption / speak-immediately policy
  (Live has no ``semantic_vad`` eagerness flag — prompting is the lever).
* Finalizes faster once mid-stream transcript already arrived (less waiting
  for a post-EOA second answer).
* Still receives ``session.output_transcript.delta`` **while** audio is being
  appended; those tokens keep their arrival timestamps for eval.

Use with runner model id containing ``duplex``, e.g.::

    --model gpt-live-duplex

The Azure/OpenAI deployment name still comes from ``GPT_DEPLOYMENT_NAME``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Optional

from realqa.adapters.openai_live import (
    DEFAULT_VOICE,
    OpenAILiveClient,
)
from realqa.base import ContentDoneEvent, SessionConfig, SessionEndEvent


# Prompt wrapper aligned with OpenAI Live prompting guide:
# https://developers.openai.com/api/docs/guides/live-prompting
# Keeps required Backchannel / Interruption / Delegation policy headings;
# borrows the "as they arrive" timing from the translation appendix for
# streaming detection (speak the target as soon as it is heard).
DUPLEX_INSTRUCTION_WRAPPER = """\
You are a precise realtime voice assistant for streaming speech understanding.
Speak clearly and briefly. Do not narrate, apologize, or explain your role.

Backchannel policy: Do not use mm-hmm or other listening sounds. When the \
task target appears, speak that target word or short phrase instead of a \
backchannel.

Interruption policy: You may speak while the user is still talking. Stop or \
yield only if the user clearly cuts you off with new speech that overrides \
the prior target. Prefer overlapping with continuous read speech so you can \
answer as soon as the target is heard — do not wait for end-of-sentence or \
silence.

Delegation policy:
Backend tools: none

Delegate to the backend when:
- Never. There is no backend for this session.

Do not delegate to the backend when:
- Always. Handle the task with voice only.

Delegate before giving an answer that depends on backend work.
Do not guess the result while waiting.

Streaming answer rules (like phrase-as-they-arrive translation):
- Treat the incoming audio as a live stream, not a finished recording.
- Respond as phrases arrive: the moment you hear the target from the task \
instructions below, speak that target immediately.
- One target occurrence → say it once; if it repeats, say it again only for \
new occurrences.
- After brief pauses, continue listening from the next unheard words; never \
restart the whole clip in your head and delay the answer.
- Output only what the task asks for (usually a single word or short phrase). \
No preamble, no full-sentence paraphrase of the passage.

Task instructions:
{task_prompt}
"""

# Faster settle once we already got mid-stream text; still wait a bit if empty.
DUPLEX_IDLE_AFTER_TRANSCRIPT_S = 0.6
DUPLEX_IDLE_NO_TRANSCRIPT_S = 6.0


class OpenAILiveDuplexClient(OpenAILiveClient):
    """Per-utterance Live client with duplex-oriented prompting + finalize."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
    ):
        super().__init__(api_key=api_key, base_url=base_url, voice=voice)

    @staticmethod
    def _resolve_model(config_model: Optional[str]) -> str:
        """Map eval aliases like ``gpt-live-duplex`` to the plain model id.

        Deployment-name substitution is Azure-specific and handled by the
        parent connector once the effective endpoint is known."""
        model = (config_model or "").strip()
        ml = model.lower()
        if not model or "duplex" in ml or ml in ("gpt-live", "gpt-live-1"):
            return "gpt-live-1"
        return model

    @staticmethod
    def _wrap_instructions(task_prompt: str) -> str:
        task = (task_prompt or "").strip() or (
            "Listen to the audio and follow the user's task."
        )
        return DUPLEX_INSTRUCTION_WRAPPER.format(task_prompt=task)

    async def _async_connect(self, config: SessionConfig) -> None:
        original_prompt = config.prompt
        original_model = config.model
        try:
            config.prompt = self._wrap_instructions(original_prompt or "")
            config.model = self._resolve_model(original_model)
            await super()._async_connect(config)
        finally:
            config.prompt = original_prompt
            config.model = original_model

    async def _finalize(self) -> None:
        """Same lifecycle as parent, shorter idle once mid-stream text exists."""
        assert self._ws is not None
        idle_ok = DUPLEX_IDLE_AFTER_TRANSCRIPT_S
        idle_empty = DUPLEX_IDLE_NO_TRANSCRIPT_S
        t_wait0 = time.time()
        while True:
            await asyncio.sleep(0.1)
            with self._lock:
                got = self._got_text_delta
                last = self._last_transcript_ts
                closed = self._session_closed
            if closed:
                return
            if got and last > 0 and (time.time() - last) >= idle_ok:
                break
            if not got and (time.time() - t_wait0) >= idle_empty:
                break
            if (time.time() - t_wait0) >= 45.0:
                break

        with self._lock:
            if self._session_closed:
                return
            self._closing = True
        await self._ws.send(json.dumps({
            "type": "session.close",
            "event_id": f"close_{uuid.uuid4().hex[:10]}",
        }))
        for _ in range(150):
            await asyncio.sleep(0.1)
            with self._lock:
                if self._session_closed:
                    return
        if not self._session_closed:
            ts = self._now()
            if self._got_text_delta:
                self._event_queue.put(ContentDoneEvent(ts, ""))
            self._event_queue.put(SessionEndEvent(ts))
