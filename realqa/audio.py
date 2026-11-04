"""Audio loading and preprocessing utilities.

Reads FLAC/WAV files, resamples, and converts to PCM16 bytes suitable
for streaming API consumption.
"""

from __future__ import annotations

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore


class AudioLoader:
    """Load, validate, resample, and convert audio files to PCM16.

    Parameters
    ----------
    target_sample_rate : int
        Desired output sample rate (Hz).  Default 16000.
    chunk_ms : int
        Chunk duration in milliseconds for streaming.  Default 200.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_ms: int = 200,
    ):
        self.target_sample_rate = target_sample_rate
        self.chunk_ms = chunk_ms
        self._chunk_samples = int(target_sample_rate * chunk_ms / 1000)

    @property
    def chunk_size(self) -> int:
        """Number of samples per streaming chunk at the target rate."""
        return self._chunk_samples

    @property
    def chunk_bytes(self) -> int:
        """Number of bytes per streaming chunk (2 bytes per int16 sample)."""
        return self._chunk_samples * 2

    def load(self, path: str) -> tuple[np.ndarray, int]:
        """Read an audio file and return (float32_mono_samples, original_sample_rate)."""
        if sf is None:
            raise ImportError("soundfile is required for audio loading")
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample *audio* to ``target_sample_rate`` if rates differ."""
        if orig_sr == self.target_sample_rate:
            return audio
        from scipy import signal as scipy_signal
        num_samples = int(len(audio) * self.target_sample_rate / orig_sr)
        return scipy_signal.resample(audio, num_samples)

    def to_pcm16(self, audio: np.ndarray) -> bytes:
        """Normalise float32 audio to [-1, 1], scale to int16, return raw bytes."""
        peak = max(abs(audio).max(), 1e-8)
        scaled = (audio / peak * 32767).astype(np.int16)
        return scaled.tobytes()

    def chunkify(self, pcm_bytes: bytes) -> list[bytes]:
        """Split PCM16 bytes into fixed-size chunks for streaming."""
        cb = self.chunk_bytes
        return [
            pcm_bytes[i : i + cb]
            for i in range(0, len(pcm_bytes), cb)
        ]

    def load_for_streaming(self, path: str) -> tuple[list[bytes], float]:
        """Convenience: load, resample, convert, chunkify.

        Returns
        -------
        chunks : list[bytes]
            PCM16 chunks ready for ``send_audio``.
        duration_s : float
            Audio duration in seconds at the target sample rate.
        """
        audio, sr = self.load(path)
        audio = self.resample(audio, sr)
        pcm = self.to_pcm16(audio)
        chunks = self.chunkify(pcm)
        duration = len(audio) / self.target_sample_rate
        return chunks, duration
