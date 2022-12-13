import json
import os
from torch.utils.data import Dataset
import torchaudio
import random
import math

class AudioDataset(Dataset):
    def __init__(self, database_path, target_sample_rate=10_000, chunk_duration=1, train_mode=True, train_test_split=0.8):
        self.database = self.load_database(database_path, train_mode, train_test_split)
        self.folder = os.path.dirname(database_path)
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        # dict of file name to tuple of waveform and sample rate
        self.audio = {filename: self.resample(*torchaudio.load(os.path.join(self.folder, filename))) for filename in self.database.keys()}

    def load_database(self, database_path, train_mode, train_test_split):
        database = json.load(open(database_path, 'r'))
        split_index = math.floor(len(database['audioFiles']) * train_test_split)

        shuffled_files = database['audioFiles'].copy()
        random.shuffle(shuffled_files)

        if train_mode:
            files = shuffled_files[:split_index]
        else:
            files = shuffled_files[split_index:]

        return {audio_file["path"]: audio_file["channels"] for audio_file in files}

    def resample(self, waveform, sample_rate):
        channel = 0
        return torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform[channel,:].view(1, -1))

    def __len__(self):
        total_samples = 0
        for waveform in self.audio.values():
            total_samples += waveform.shape[1]
        duration = total_samples // self.target_sample_rate
        return duration // self.chunk_duration
    
    def get_whistle_labels(self, filename):
        file_data = self.database[filename]
        return file_data[0]['whistleLabels']
    
    def get_label(self, filename, start):
        end = start + self.chunk_duration * self.target_sample_rate
        for label in self.get_whistle_labels(filename):
            if not (label['start'] < start and label['end'] < start \
                or label['start'] > end and label['end'] > end):
                return True
        return False

    def __getitem__(self, _):
        filename = random.choice(list(self.audio.keys()))
        waveform = self.audio[filename]
        start_pos = random.randint(0, waveform.shape[1] - self.target_sample_rate * self.chunk_duration)
        chunk = waveform[:, start_pos:start_pos + self.target_sample_rate * self.chunk_duration]
        label = self.get_label(filename, start_pos)

        # Repeat the mel spectrogram 3 times to match the number of channels to the expected input of the model
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(self.target_sample_rate, n_mels=64)(chunk) \
            .log2() \
            .repeat(3, 1, 1)


        return mel_spectrogram, label
