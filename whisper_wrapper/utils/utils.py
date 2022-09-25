import os
import sys

import librosa


def read_audio_in_batch(file_path, seconds):
    # Read audio file in batches
    waveform, sample_rate = librosa.load(file_path)

    TOTAL_DURATION = waveform.shape[0] // (sample_rate)
    if seconds > TOTAL_DURATION:
        raise ValueError(f"seconds should be less than {TOTAL_DURATION}")

    step_size = sample_rate * seconds

    return [waveform[i : i + step_size] for i in range(0, waveform.shape[0], step_size)]


def blockPrint():
    # Disable console logs
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    # Enable console logs
    sys.stdout = sys.__stdout__
