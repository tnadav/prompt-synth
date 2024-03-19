import math

import librosa
import numpy as np
import numpy.typing as npt

from .note_gen import NoteGenerator
from .pitch_shift import pitch_shift


def _normalize(audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    audio /= np.max(np.abs(audio))
    return audio


class SampleNoteGenerator(NoteGenerator):
    @classmethod
    def from_sample(
        cls,
        sample: npt.NDArray[np.float64],
        base_freq: float,
        sample_rate: int,
        generator_sample_rate: int | None = None,
    ) -> "SampleNoteGenerator":
        generator_sample_rate = (
            sample_rate if generator_sample_rate is None else generator_sample_rate
        )

        result = cls(generator_sample_rate)
        result.set_sample(sample, sample_rate, base_freq)
        return result

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self._base_freq = 440.0
        self._note_cache: dict[float, npt.NDArray[np.float64]] = {440.0: np.array([])}

    @property
    def _sample(self) -> npt.NDArray[np.float64]:
        return self._note_cache[self._base_freq]

    def set_sample(
        self, sample: npt.NDArray[np.float64], sample_rate: int, base_freq: float
    ) -> None:
        if sample_rate != self.sample_rate:
            sample = librosa.resample(
                sample, orig_sr=sample_rate, target_sr=self.sample_rate
            )

        self._base_freq = base_freq
        self._note_cache = {base_freq: _normalize(sample)}

    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        note_sample = self._note_cache.get(freq)
        if note_sample is None:
            note_sample = pitch_shift(
                self._sample, self.sample_rate, self._base_freq, freq
            )
            self._note_cache[freq] = note_sample

        length_idx = int(math.ceil(duration * self._sample_rate))
        return note_sample[:length_idx]
