"""Main module."""
from typing import Union

import torchaudio
import whisper
from tqdm import tqdm

from whisper_wrapper.utils import utils


def transcribe_in_batches(
    file_path: str,
    seconds: int = 10,
    model_name: Union["tiny", "base", "small", "medium", "large"] = "base",
    show_print: bool = False,
):
    """Transcribe audio file in batches.

    Args:
        file_path (str): Path to audio file.
        seconds (int): Seconds to transcribe.
        model_name (str): Model name to use for transcription.

    Returns:
        list: List of transcription results.
    """
    audio_batches = utils.read_audio_in_batch(file_path, seconds)
    results = []
    for audio_batch in audio_batches:
        if show_print:
            utils.blockPrint()
        model = whisper.load_model(model_name)
        results.append(model.transcribe(audio_batch))
        if show_print:
            utils.enablePrint()
