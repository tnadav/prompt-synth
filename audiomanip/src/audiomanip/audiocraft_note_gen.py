from enum import Enum

from audiocraft.models import MAGNeT, MusicGen  # type: ignore [attr-defined]

from .sampler import SampleNoteGenerator

_N_VARIATIONS = 3


class ModelType(Enum):
    MusicGen = 1
    MAGNeT = 2


class PromptNoteGenerator:
    def __init__(self, model: MusicGen | MAGNeT):
        self._model = model

    def from_prompt(self, prompt: str) -> SampleNoteGenerator:
        descriptions = [prompt for _ in range(_N_VARIATIONS)]
        output = self._model.generate(
            descriptions=descriptions, progress=True, return_tokens=True
        )
        samples = output[0].detach().cpu()
        if samples.dim() == 2:
            samples = samples[None, ...]

        return SampleNoteGenerator.from_sample(
            samples[0][0].numpy(), 440.0, self._model.compression_model.sample_rate
        )


def make_audiocraft_note_generator(
    model_type: ModelType, state_dir: str
) -> PromptNoteGenerator:
    if model_type == ModelType.MusicGen:
        model = MusicGen.get_pretrained(state_dir)
    elif model_type == ModelType.MAGNeT:
        model = MAGNeT.get_pretrained(state_dir)
    else:
        raise ValueError(f"Unsupported model type {model_type}")

    return PromptNoteGenerator(model)
