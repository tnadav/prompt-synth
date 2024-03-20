import json
import os
from typing import Iterator, NamedTuple

import librosa


class NSynthData(NamedTuple):
    base_directory: str
    pitch: int
    velocity: int
    sample_rate: int
    qualities: list[str]
    instrument_id: int
    instrument_source: str
    instrument_family: str
    note: str

    @property
    def name(self) -> str:
        prefix = ""
        if "bright" in self.qualities:
            prefix = "bright "
        elif "dark" in self.qualities:
            prefix = "dark "

        suffix = ""
        if "reverb" in self.qualities:
            suffix = " with reverb"
        elif "distortion" in self.qualities:
            suffix = " with distortion"

        return f"{prefix}{self.instrument_source} {self.instrument_family}{suffix}"

    @property
    def wav_path(self) -> str:
        return os.path.join(self.base_directory, "audio", f"{self.note}.wav")

    @property
    def base_freq(self) -> float:
        return float(librosa.midi_to_hz(self.pitch))


class NSynthDataset:
    def __init__(self, directory: str):
        self._base_dir = directory
        with open(os.path.join(directory, "examples.json"), "r") as f:
            self._examples = json.load(f)

    def __iter__(self) -> Iterator[NSynthData]:
        for example in self._examples.values():
            yield NSynthData(
                self._base_dir,
                example["pitch"],
                example["velocity"],
                example["sample_rate"],
                example["qualities_str"],
                example["instrument"],
                example["instrument_source_str"],
                example["instrument_family_str"],
                example["note_str"],
            )
