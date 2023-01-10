import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torchaudio

from .dataset import convert_waveform_to_spectogram, resample
from .model import get_model
from .utils import print_environment_info


def _detect(
    waveform: torch.Tensor,
    sample_rate: int,
    weights_path: os.PathLike,
    conf_thresh: float,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    spectogram = convert_waveform_to_spectogram(sample_rate, waveform)
    spectogram = spectogram.to(device)
    output = model(spectogram).squeeze()

    print(output)


def detect_directory(
    input_directory: os.PathLike,
    weights_path: os.PathLike,
    sample_rate: int,
    chunk_duration: int,
    conf_thresh: float,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detects whistles in all audio files in the given directory.

    :param input_directory: Path to the directory with audio files
    :type input_directory: os.PathLike
    :param weights_path: Path to the weights file
    :type weights_path: os.PathLike
    :param sample_rate: Sample rate to use for the audio files
    :type sample_rate: int
    :param chunk_duration: Duration of the chunks to use for detection
    :type chunk_duration: int
    :param conf_thresh: Confidence threshold for the detection
    :type conf_thresh: float
    :return: Dictionary with the audio file as key and a list of tuples with the start and end of the detected whistle
    :rtype: Dict[str, List[Tuple[int, int]]]
    """
    wav_files = glob.glob(f"{input_directory}/*.wav")
    result: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for wav_file in wav_files:
        waveform, original_sample_rate = torchaudio.load(wav_file)
        waveform = resample(waveform, original_sample_rate, sample_rate)

        print(waveform.shape)

        for sample in range(0, waveform.shape[1], sample_rate * chunk_duration):
            chunk = waveform[:, sample : sample + sample_rate * chunk_duration]
            if _detect(chunk, sample_rate, weights_path, conf_thresh):
                result[wav_file].append((sample, sample + sample_rate * chunk_duration))

    return result


def _run():
    print_environment_info()

    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="checkpoints/checkpoint.pth",
        help="Path to weights or checkpoint file (.pth)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/samples",
        help="Path to directory with audio files to classify",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.5, help="Whistle confidence threshold"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=10_000,
        help="Targeted sample rate of the audio files",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=1,
        help="Duration of the chunks in seconds",
    )
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    detect_directory(args.input, args.weights, args.sample_rate, args.chunk_duration, args.conf_thres)


if __name__ == "__main__":
    _run()
