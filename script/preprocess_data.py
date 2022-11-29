import json
import wave
import os
from pathlib import Path

folder = 'data'
processed_folder = 'processed'

required_folders = [f'{processed_folder}/whistle', f'{processed_folder}/no_whistle']
for required_folder in required_folders:
    if not os.path.exists(required_folder):
        os.makedirs(required_folder, exist_ok=True)

SAMPLE_SIZE = 4096

database = json.load(open(f'{folder}/whistledb.json', 'r'))
audio_files = [Path(f'{folder}/{file_data["path"]}') for file_data in database['audioFiles']]

def get_whistle_labels(audio_file):
    file_data = next(file for file in database['audioFiles'] if file['path'] == str(audio_file))
    return file_data['channels'][0]['whistleLabels']

def contains_whistle(filename, start):
    end = start + SAMPLE_SIZE
    whistles = get_whistle_labels(filename)
    for whistle in whistles:
        if whistle['start'] <= start and whistle['end'] >= end:
            return True
    return False

def store_frames(wav, filename, start, label):
    wav.setpos(start)
    frames = wav.readframes(SAMPLE_SIZE)
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()
    channels = wav.getnchannels()
    with wave.open(f'{processed_folder}/{label}/{filename}_{start}.wav', 'wb') as out_wav:
        out_wav.setnchannels(channels)
        out_wav.setsampwidth(sample_width)
        out_wav.setframerate(sample_rate)
        out_wav.writeframes(frames)

for audio_file in audio_files:
    with wave.open(str(audio_file), 'rb') as wav:


        nframes = wav.getnframes()
        for i in range(0, nframes, SAMPLE_SIZE):
            if i + SAMPLE_SIZE > nframes:
                break

            if contains_whistle(audio_file.name, i):
                store_frames(wav, audio_file.stem, i, 'whistle')
            else:
                store_frames(wav, audio_file.stem, i, 'no_whistle')
