"""Adapter registry for streaming API clients.

Imports are intentionaly lazy — each adapter has optional heavy deps
(dashscope / google-genai).  Import the concrete module you need::

    from realqa.adapters.gemini_live import GeminiLiveClient
"""

from __future__ import annotations

__all__ = [
    "QwenOmniRealtimeClient",
    "QwenAudioRealtimeClient",
    "GeminiLiveClient",
    "OpenAIRealtimeClient",
    "OpenAILiveClient",
    "OpenAILiveDuplexClient",
]


def __getattr__(name: str):
    if name == "QwenOmniRealtimeClient":
        from realqa.adapters.qwen_omni import QwenOmniRealtimeClient
        return QwenOmniRealtimeClient
    if name == "QwenAudioRealtimeClient":
        from realqa.adapters.qwen_audio_realtime import QwenAudioRealtimeClient
        return QwenAudioRealtimeClient
    if name == "GeminiLiveClient":
        from realqa.adapters.gemini_live import GeminiLiveClient
        return GeminiLiveClient
    if name == "OpenAIRealtimeClient":
        from realqa.adapters.openai_realtime import OpenAIRealtimeClient
        return OpenAIRealtimeClient
    if name == "OpenAILiveClient":
        from realqa.adapters.openai_live import OpenAILiveClient
        return OpenAILiveClient
    if name == "OpenAILiveDuplexClient":
        from realqa.adapters.openai_live_duplex import OpenAILiveDuplexClient
        return OpenAILiveDuplexClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
