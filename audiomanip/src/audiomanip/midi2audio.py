import math
from typing import IO, Any, Iterable, NamedTuple, cast

import mido
import numpy as np
import numpy.typing as npt

from .note_gen import NoteGenerator, SineOscNoteGenerator

DEFAULT_TEMPO = 500000
RELEASE_SAMPLES = 44
RELEASE_SECONDS = 0.001


class NoteEvent(NamedTuple):
    time: float
    duration: float
    velocity: int
    note: int


def midi_to_note_events(midi: mido.MidiFile) -> Iterable[NoteEvent]:
    pending_notes: dict[int, tuple[float, Any]] = {}

    tempo = DEFAULT_TEMPO
    last_time = 0.0

    def tick2second(ticks: int) -> float:
        return cast(float, mido.tick2second(ticks, midi.ticks_per_beat, tempo))

    for track in midi.tracks:
        cur_tick = 0
        for msg in track:
            cur_tick += msg.time
            time = tick2second(cur_tick)

            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on":
                if msg.note not in pending_notes:
                    pending_notes[msg.note] = (time, msg)
            elif msg.type == "note_off":
                note_on = pending_notes.pop(msg.note, None)
                if note_on is not None:
                    start_time, note_on_msg = note_on
                    yield NoteEvent(
                        start_time, time - start_time, note_on_msg.velocity, msg.note
                    )
                last_time = max(last_time, time)

    for (start_time, msg) in pending_notes.values():
        yield NoteEvent(start_time, last_time - start_time, msg.velocity, msg.note)


def midi2audio(
    midi_file: IO[bytes], note_gen: NoteGenerator | None = None
) -> npt.NDArray[np.float64]:
    note_gen = SineOscNoteGenerator(44_100) if note_gen is None else note_gen
    sample_rate = note_gen.sample_rate

    release_samples = int(math.ceil(sample_rate * RELEASE_SECONDS))
    midi = mido.MidiFile(file=midi_file)

    audio = np.zeros(
        (int(math.ceil(sample_rate * midi.length))),
    )

    for msg in midi_to_note_events(midi):
        # Calculate the frequency of the note
        freq = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))

        # Generate the audio for the note
        note = note_gen.gen_note(freq, msg.duration)
        env = np.concatenate(
            (
                np.ones(len(note) - release_samples),
                np.linspace(1, 0, release_samples),
            )
        )

        # Scale the note by the velocity
        note *= msg.velocity / 127.0
        note *= env

        start_index = int(msg.time * sample_rate)
        # Add the note to the audio
        audio[start_index : start_index + note.shape[0]] += note

    # Normalize the audio
    audio /= np.max(np.abs(audio))

    return audio
