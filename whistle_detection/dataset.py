import json
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

# folder = 'data'
# processed_folder = 'processed'

# required_folders = [f'{processed_folder}/whistle', f'{processed_folder}/no_whistle']
# for required_folder in required_folders:
#     if not os.path.exists(required_folder):
#         os.makedirs(required_folder, exist_ok=True)

# SAMPLE_SIZE = 4096

# audio_files = [Path(f'{folder}/{file_data["path"]}') for file_data in database['audioFiles']]

# def get_whistle_labels(audio_file):
#     file_data = next(file for file in database['audioFiles'] if file['path'] == str(audio_file))
#     return file_data['channels'][0]['whistleLabels']

# def contains_whistle(filename, start):
#     end = start + SAMPLE_SIZE
#     whistles = get_whistle_labels(filename)
#     for whistle in whistles:
#         if whistle['start'] <= start and whistle['end'] >= end:
#             return True
#     return False

# def store_frames(wav, filename, start, label):
#     wav.setpos(start)
#     frames = wav.readframes(SAMPLE_SIZE)
#     sample_rate = wav.getframerate()
#     sample_width = wav.getsampwidth()
#     channels = wav.getnchannels()
#     with wave.open(f'{processed_folder}/{label}/{filename}_{start}.wav', 'wb') as out_wav:
#         out_wav.setnchannels(channels)
#         out_wav.setsampwidth(sample_width)
#         out_wav.setframerate(sample_rate)
#         out_wav.writeframes(frames)

# for audio_file in audio_files:
#     with wave.open(str(audio_file), 'rb') as wav:


#         nframes = wav.getnframes()
#         for i in range(0, nframes, SAMPLE_SIZE):
#             if i + SAMPLE_SIZE > nframes:
#                 break

#             if contains_whistle(audio_file.name, i):
#                 store_frames(wav, audio_file.stem, i, 'whistle')
#             else:
#                 store_frames(wav, audio_file.stem, i, 'no_whistle')

class AudioDataset(Dataset):
    def __init__(self, database_path, target_sample_rate=10_000, chunk_duration=1):
        database = json.load(open(database_path, 'r'))
        self.folder = os.path.dirname(database_path)
        self.files = [os.path.join(self.folder, file_data['path']) for file_data in database['audioFiles']]
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        # dict of file name to tuple of waveform and sample rate
        self.audio = {audio_file: self.resample(*torchaudio.load(audio_file)) for audio_file in self.files}

    def resample(self, waveform, sample_rate):
        channel = 0
        return torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform[channel,:].view(1, -1))

    def __len__(self):
        total_samples = 0
        for audio_file in self.audio.items():
            total_samples += audio_file.shape[1]
        duration = total_samples / self.target_sample_rate
        return duration // self.chunk_duration

    # TODO: Implement later
    def __getitem__(self, idx):
        file = self.files[idx]
        with wave.open(f'{self.folder}/{file}', 'rb') as wav:
            frames = wav.readframes(SAMPLE_SIZE)
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            channels = wav.getnchannels()
            return frames, sample_rate, sample_width, channels
