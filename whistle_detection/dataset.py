import glob
import json
import math
import os
import random
from typing import Dict, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


class DirectoryDataset(Dataset):
    def __init__(
        self,
        directory: os.PathLike,
        target_sample_rate: Optional[int] = 10_000,
        chunk_duration: Optional[float] = 1,
    ) -> None:
        """Simple dataset to load a directory of audio .wav files.

        :param directory: Path to the directory with audio files
        :type directory: os.PathLike
        :param target_sample_rate: Target sample rate for the audio files, defaults to 10_000
        :type target_sample_rate: Optional[int], optional
        :param chunk_duration: Duration of the chunks (in seconds), defaults to 1
        :type chunk_duration: Optional[float], optional
        """
        self.directory: os.PathLike = directory
        self.target_sample_rate: int = target_sample_rate
        self.audiofiles: Dict[os.PathLike, Tuple[torch.Tensor, int]] = {
            filename: torchaudio.load(filename)
            for filename in glob.glob(f"{directory}/*.wav")
        }
        # Resample all audio files to the target sample rate
        self.audio = {
            filename: resample(waveform, sample_rate, target_sample_rate)
            for filename, (waveform, sample_rate) in self.audiofiles.items()
        }

    def __len__(self) -> int:
        total_samples = 0
        for waveform in self.audio.values():
            total_samples += waveform.shape[1]
        duration = total_samples // self.target_sample_rate
        return duration // self.chunk_duration

    def _get_filename_and_chunk_offset_by_index(
        self, idx: int
    ) -> Tuple[os.PathLike, int]:
        """
        Get the filename and chunk offset by index.

        :param idx: Index of the item
        :type idx: int
        :return: Tuple of filename and chunk offset in samples in the file
        :rtype: Tuple[os.PathLike, int]
        """
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")

        chuck_samples = self.target_sample_rate * self.chunk_duration
        skipped_samples = 0
        for filename, (waveform, _) in self.audio.items():
            # Check, if the index is in the current file
            if skipped_samples + waveform.shape[1] > idx * chuck_samples:
                return filename, idx * chuck_samples - skipped_samples
            else:
                skipped_samples += waveform.shape[1]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, int, os.PathLike, int, int]:
        """
        Get a MEL-spectrogram chunk of audio from the dataset.

        :param idx: Index of the item
        :type idx: int
        :return: Tuple of MEL-spectrogram, sample rate, original sample rate, filename, start, end
        :rtype: Tuple[torch.Tensor, int, int, os.PathLike, int, int]
        """
        filename: os.PathLike
        chunk_start: int
        filename, chunk_start = self._get_filename_and_chunk_offset_by_index(idx)
        chunk: torch.Tensor = self.audio[filename][
            :, chunk_start : chunk_start + self.target_sample_rate * self.chunk_duration
        ]
        mel_spectrogram: torch.Tensor = convert_waveform_to_spectogram(
            self.target_sample_rate, chunk
        )

        original_sample_rate: int = self.audiofiles[filename][1]
        return (
            mel_spectrogram,
            self.target_sample_rate,
            original_sample_rate,
            filename,
            chunk_start,
            chunk_start + self.target_sample_rate * self.chunk_duration,
        )


class AudioDataset(Dataset):
    def __init__(
        self,
        database_path: os.PathLike,
        target_sample_rate: Optional[int] = 10_000,
        chunk_duration: Optional[float] = 1,
        train_mode: Optional[bool] = True,
        train_test_split: Optional[float] = 0.8,
        seed: Optional[int] = -1,
    ):
        """
        Dataset for the audio files and labeles in the BHuman format.

        :param database_path: Path to the database file (.json)
        :type database_path: os.PathLike
        :param target_sample_rate: Target sample rate for the audio files, defaults to 10_000
        :type target_sample_rate: Optional[int], optional
        :param chunk_duration: Duration of the chunks (in seconds), defaults to 1
        :type chunk_duration: Optional[float], optional
        :param train_mode: Whether to use the training or test set, defaults to True
        :type train_mode: Optional[bool], optional
        :param train_test_split: Percentage of the data to use for training, defaults to 0.8
        :type train_test_split: Optional[float], optional
        :param seed: Seed for the random number generator, defaults to -1
        :type seed: Optional[int], optional
        """
        random.seed(seed)
        self.database = self.load_database(database_path, train_mode, train_test_split)
        self.folder = os.path.dirname(database_path)
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        # dict of file name to tuple of waveform and sample rate
        self.audio = {
            filename: resample(
                *torchaudio.load(os.path.join(self.folder, filename)),
                target_sample_rate,
            )
            for filename in self.database.keys()
        }

    def load_database(
        self, database_path, train_mode, train_test_split
    ) -> Dict[str, Any]:
        database = json.load(open(database_path, "r"))
        split_index = math.floor(len(database["audioFiles"]) * train_test_split)

        shuffled_files = database["audioFiles"].copy()
        random.shuffle(shuffled_files)

        if train_mode:
            files = shuffled_files[:split_index]
        else:
            files = shuffled_files[split_index:]

        return {audio_file["path"]: audio_file["channels"] for audio_file in files}

    def __len__(self) -> int:
        total_samples = 0
        for waveform in self.audio.values():
            total_samples += waveform.shape[1]
        duration = total_samples // self.target_sample_rate
        return duration // self.chunk_duration

    def get_whistle_labels(self, filename: str) -> List[Dict[str, int]]:
        """
        Get the whistle labels for a file.

        :param filename: Name of the file
        :type filename: str
        :return: List of whistle labels (start and end in samples)
        :rtype: List[Dict[str, int]]
        """
        return self.database[filename][0]["whistleLabels"]

    def get_label(self, filename: str, start: int) -> bool:
        """
        Get the label for a chunk of audio.

        :param filename: Name of the file
        :type filename: str
        :param start: Start of the chunk in samples
        :type start: int
        :return: True if the chunk contains a whistle
        :rtype: bool
        """
        end = start + self.chunk_duration * self.target_sample_rate
        for label in self.get_whistle_labels(filename):
            if not (
                label["start"] < start
                and label["end"] < start
                or label["start"] > end
                and label["end"] > end
            ):
                return True
        return False

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, bool]:
        """
        Get a MEL-spectrogram of audio random chunk from the dataset.

        :param idx: Index of the item
        :type idx: int
        :return: Tuple of MEL-spectrogram and label (True if whistle is present)
        :rtype: Tuple[torch.Tensor, bool]
        """
        random.seed(idx)
        filename: os.PathLike = random.choice(list(self.audio.keys()))
        waveform: torch.Tensor = self.audio[filename]
        start_pos: int = random.randint(
            0, waveform.shape[1] - self.target_sample_rate * self.chunk_duration
        )
        chunk: torch.Tensor = waveform[
            :, start_pos : start_pos + self.target_sample_rate * self.chunk_duration
        ]
        label: bool = self.get_label(filename, start_pos)

        mel_spectrogram: torch.Tensor = convert_waveform_to_spectogram(
            self.target_sample_rate, chunk
        )
        return mel_spectrogram, label


def convert_waveform_to_spectogram(
    sample_rate: int, chunk: torch.Tensor
) -> torch.Tensor:
    """
    Converts a waveform to a mel spectrogram.

    :param sample_rate: Sample rate of the waveform
    :type sample_rate: int
    :param chunk: Waveform
    :type chunk: torch.Tensor
    :return: Mel spectrogram
    :rtype: torch.Tensor
    """
    # Repeat the mel spectrogram 3 times to match the number of channels to the expected input of the model
    # TODO: Discuss whether to change the model to accept 1 channel
    return (
        torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=64)(chunk)
        .log2()
        .repeat(3, 1, 1)
    )


def resample(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int
) -> torch.Tensor:
    """
    Resamples a waveform to a target sample rate.

    :param waveform: Waveform
    :type waveform: torch.Tensor
    :param sample_rate: Sample rate of the waveform
    :type sample_rate: int
    :param target_sample_rate: Target sample rate
    :type target_sample_rate: int
    :return: Resampled waveform
    :rtype: torch.Tensor
    """
    channel = 0
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(
        waveform[channel, :].view(1, -1)
    )
