import argparse
import io
from contextlib import closing
from typing import Type

import numpy as np
import numpy.typing as npt
import pyaudio

from audiomanip import (
    NoteGenerator,
    SawToothOscNoteGenerator,
    SineOscNoteGenerator,
    SquareOscNoteGenerator,
    TriangleOscNoteGenerator,
    midi2audio,
)

_OSCILATORS: dict[str, Type[NoteGenerator]] = {
    "sine": SineOscNoteGenerator,
    "sawtooth": SawToothOscNoteGenerator,
    "square": SquareOscNoteGenerator,
    "triangle": TriangleOscNoteGenerator,
}
_CHUNK_SIZE = 1024


def play_audio(buffer: npt.NDArray[np.float64], sample_rate: int) -> None:
    audio = buffer.astype(np.float32)

    p = pyaudio.PyAudio()
    try:
        buf = io.BytesIO(audio.tobytes())
        with closing(
            p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
        ) as stream:
            while len(data := buf.read(_CHUNK_SIZE)):
                stream.write(data)
    finally:
        p.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a MIDI file.")
    parser.add_argument("midi_file", type=str, help="The MIDI file to play.")
    parser.add_argument(
        "--osc",
        choices=_OSCILATORS.keys(),
        default="sine",
        help="The MIDI file to play.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=44100, help="The sample rate of the audio."
    )
    args = parser.parse_args()

    note_gen = _OSCILATORS[args.osc](args.sample_rate)
    with open(args.midi_file, "rb") as f:
        audio = midi2audio(f, note_gen)

    play_audio(audio, args.sample_rate)


if __name__ == "__main__":
    main()
