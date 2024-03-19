from abc import ABC, abstractmethod
from typing import Type, cast

import numpy as np
import numpy.typing as npt


class NoteGenerator(ABC):
    def __init__(self, sample_rate: int):
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @abstractmethod
    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        pass


def _normalize(audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    audio /= np.max(np.abs(audio))
    return audio


def _sine(
    frequency: float, duration: float, sample_rate: int = 44100
) -> npt.NDArray[np.float64]:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.sin(2 * np.pi * frequency * t)
    return cast(npt.NDArray[np.float64], signal)


def _bl_square(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    max_harmonic: int = 10000,
) -> npt.NDArray[np.float64]:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.zeros_like(t)

    # Sum over odd harmonics up to the Nyquist frequency
    harmonic = frequency
    while harmonic < max_harmonic:
        signal += np.sin(2 * np.pi * harmonic * t) / harmonic
        harmonic += 2 * frequency

    return _normalize(signal)


def _bl_triangle(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    max_harmonic: int = 10000,
) -> npt.NDArray[np.float64]:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.zeros_like(t)

    # Sum over odd harmonics up to the Nyquist frequency
    harmonic = frequency
    n = 1
    while harmonic < max_harmonic:
        signal += (-1) ** ((n - 1) / 2) * np.sin(2 * np.pi * harmonic * t) / (n**2)
        harmonic += 2 * frequency
        n += 2

    return _normalize(signal)


def _bl_sawtooth(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    max_harmonic: int = 10000,
) -> npt.NDArray[np.float64]:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.zeros_like(t)

    # Sum over harmonics up to the Nyquist frequency
    harmonic = frequency
    while harmonic < max_harmonic:
        signal += np.sin(2 * np.pi * harmonic * t) / harmonic
        harmonic += frequency

    return _normalize(signal)


class SineOscNoteGenerator(NoteGenerator):
    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        return _sine(freq, duration, self._sample_rate)


class SawToothOscNoteGenerator(NoteGenerator):
    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        return _bl_sawtooth(freq, duration, self._sample_rate, self._sample_rate // 2)


class SquareOscNoteGenerator(NoteGenerator):
    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        return _bl_square(freq, duration, self._sample_rate, self._sample_rate // 2)


class TriangleOscNoteGenerator(NoteGenerator):
    def gen_note(self, freq: float, duration: float) -> npt.NDArray[np.float64]:
        return _bl_triangle(freq, duration, self._sample_rate, self._sample_rate // 2)


OSCILATORS: dict[str, Type[NoteGenerator]] = {
    "sine": SineOscNoteGenerator,
    "sawtooth": SawToothOscNoteGenerator,
    "square": SquareOscNoteGenerator,
    "triangle": TriangleOscNoteGenerator,
}


def make_osc(shape: str, sample_rate: int = 44_100) -> NoteGenerator:
    return OSCILATORS[shape](sample_rate)
