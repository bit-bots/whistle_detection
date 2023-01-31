import argparse
import os
from typing import Dict, List, Tuple
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from whistle_detection.dataset import DirectoryDataset
from whistle_detection.model import get_model
from whistle_detection.utils import print_environment_info


def detect(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    conf_thresh: float,
):
    for (spectrogram, sample_rate, original_sample_rate, filename, start, end) in tqdm.tqdm(dataloader, desc="Detecting"):
        spectrogram = Variable(spectrogram, requires_grad=False).to(device)

        with torch.no_grad():
            output = torch.sigmoid(model(spectrogram)).squeeze()
        
        whistles = output >= conf_thresh
        for i, whistle in enumerate(whistles):
            if whistle:
                start_time = int((start[i] / sample_rate[i]) * original_sample_rate[i])
                end_time = int((end[i] / sample_rate[i]) * original_sample_rate[i])
                print(f"{filename}: {start_time:.2f} - {end_time:.2f}")


def detect_directory(
    input_directory: os.PathLike,
    weights_path: os.PathLike,
    sample_rate: int,
    chunk_duration: int,
    conf_thresh: float,
    batch_size: int,
    n_cpu: int,
):
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
    :param batch_size: Batch size for the detection
    :type batch_size: int
    :param n_cpu: Number of CPU workers to use for the data loaders
    :type n_cpu: int
    """
    dataset = DirectoryDataset(input_directory, sample_rate, chunk_duration)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    detect(model, device, dataloader, conf_thresh)


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
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batches"
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="Number of cpu threads to use during batch generation",
    )
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    results = detect_directory(
        args.input,
        args.weights,
        args.sample_rate,
        args.chunk_duration,
        args.conf_thres,
        args.batch_size,
        args.n_cpu,
    )
    print(results)


if __name__ == "__main__":
    _run()
