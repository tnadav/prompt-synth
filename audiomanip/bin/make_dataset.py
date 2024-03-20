import argparse
import json
import os
from collections import defaultdict

import librosa
import soundfile as sf

from audiomanip import NSynthData, NSynthDataset, pitch_shift_midi_note

_STANDATD_PITCH = 69  # Set A4 as the base pitch


def sample_score(sample: NSynthData) -> float:
    pitch_dist = 2 * (sample.pitch - _STANDATD_PITCH) ** 2
    velocity_dist = (sample.velocity - 127) ** 2 / 100.0
    fast_decay_penalty = 100 if "fast_decay" in sample.qualities else 0
    long_release_penalty = 20 if "long_release" in sample.qualities else 0
    tempo_sync_penalty = 20 if "tempo_sync" in sample.qualities else 0

    return (
        pitch_dist
        + velocity_dist
        + long_release_penalty
        + fast_decay_penalty
        + tempo_sync_penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make dataset for audiocraft training")
    parser.add_argument("nsynth_path", type=str, help="NSynth wav/json dataset path")
    parser.add_argument("name", type=str, help="Output dataset name")
    parser.add_argument(
        "output_path", type=str, help="Audiocraft data set output directory"
    )

    args = parser.parse_args()
    dataset = NSynthDataset(args.nsynth_path)

    samples: dict[int, list[NSynthData]] = defaultdict(list)
    for example in dataset:
        if "percussive" in example.qualities:
            continue

        samples[example.instrument_id].append(example)

    for sample in samples.values():
        sample.sort(key=sample_score)

    os.makedirs(os.path.join(args.output_path, args.name), exist_ok=True)

    with open(os.path.join(args.output_path, "data.jsonl"), "w") as manifest:
        for name, candidates in samples.items():
            print(f"{name} ({len(candidates)}):")
            cur_sample = candidates[0]
            print(
                "\tPitch: {}\tVelocity: {}\tQualities: {}".format(
                    cur_sample.pitch, cur_sample.velocity, cur_sample.qualities
                )
            )
            audio, sample_rate = librosa.load(cur_sample.wav_path)
            if cur_sample.pitch != _STANDATD_PITCH:
                audio = pitch_shift_midi_note(
                    audio, int(sample_rate), cur_sample.pitch, _STANDATD_PITCH
                )

            wav_name = f"{cur_sample.note}.wav"
            sf.write(
                os.path.join(args.output_path, args.name, wav_name), audio, sample_rate
            )
            music_info = dict(
                title=None,
                artist=None,
                key=None,
                bpm=None,
                genre=None,
                moods=None,
                keywords=cur_sample.qualities,
                description=cur_sample.name,
                name=None,
                instrument=None,
            )
            with open(
                os.path.join(args.output_path, args.name, f"{cur_sample.note}.json"),
                "w",
            ) as info:
                json.dump(music_info, info)

            meta = dict(
                path=f"dataset/{args.name}/{wav_name}",
                duration=librosa.get_duration(y=audio, sr=sample_rate),
                sample_rate=sample_rate,
                amplitude=None,
                weight=None,
                info_path=None,
            )
            manifest_line = json.dumps(meta)
            manifest.write(f"{manifest_line}\n")


if __name__ == "__main__":
    main()
