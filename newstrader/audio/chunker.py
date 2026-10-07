"""Cut a continuous 16 kHz audio stream into short chunks for the transcriber.

Each chunk is about `chunk_seconds` long, but the cut is moved to the quietest moment in the last
1.5 seconds, so words are less likely to be split in half between two chunks.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME = int(0.03 * SAMPLE_RATE)  # 30 ms
SEARCH = int(1.5 * SAMPLE_RATE)


class AudioChunker:
    def __init__(self, chunk_seconds: float = 10.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.target = int(chunk_seconds * sample_rate)
        self._buf = np.zeros(0, dtype=np.int16)
        self._pending = b""
        self.samples_emitted = 0

    def feed(self, data: bytes) -> list[tuple[float, np.ndarray]]:
        """Add raw s16le mono bytes. Returns finished chunks as (offset_seconds, float32 audio)."""
        data = self._pending + data
        usable = len(data) - (len(data) % 2)
        self._pending = data[usable:]
        if usable:
            self._buf = np.concatenate([self._buf, np.frombuffer(data[:usable], dtype=np.int16)])
        out = []
        while len(self._buf) >= self.target:
            cut = self._quiet_cut()
            chunk = self._buf[:cut]
            self._buf = self._buf[cut:]
            offset = self.samples_emitted / self.sample_rate
            self.samples_emitted += len(chunk)
            out.append((offset, chunk.astype(np.float32) / 32768.0))
        return out

    def flush(self) -> list[tuple[float, np.ndarray]]:
        if len(self._buf) < self.sample_rate // 2:  # under half a second: drop
            self._buf = np.zeros(0, dtype=np.int16)
            return []
        chunk, self._buf = self._buf, np.zeros(0, dtype=np.int16)
        offset = self.samples_emitted / self.sample_rate
        self.samples_emitted += len(chunk)
        return [(offset, chunk.astype(np.float32) / 32768.0)]

    def _quiet_cut(self) -> int:
        end = self.target
        start = max(FRAME, end - SEARCH)
        window = self._buf[start:end].astype(np.float32)
        n_frames = len(window) // FRAME
        if n_frames < 2:
            return end
        frames = window[: n_frames * FRAME].reshape(n_frames, FRAME)
        energy = np.sqrt((frames ** 2).mean(axis=1))
        quietest = int(np.argmin(energy))
        return start + quietest * FRAME + FRAME // 2


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
