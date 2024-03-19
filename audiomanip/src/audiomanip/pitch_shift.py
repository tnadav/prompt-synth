import librosa
import numpy as np
from numpy import typing as npt


def pitch_shift(
    input_audio: npt.NDArray[np.float64],
    sample_rate: int,
    orig_base_freq: float,
    desired_base_freq: float,
) -> npt.NDArray[np.float64]:
    orig_note = int(librosa.hz_to_midi(orig_base_freq))
    desired_note = int(librosa.hz_to_midi(desired_base_freq))
    return pitch_shift_midi_note(input_audio, sample_rate, orig_note, desired_note)


def pitch_shift_midi_note(
    input_audio: npt.NDArray[np.float64],
    sample_rate: int,
    orig_note: int,
    desired_note: int,
) -> npt.NDArray[np.float64]:
    return librosa.effects.pitch_shift(
        input_audio, sr=sample_rate, n_steps=desired_note - orig_note
    )
