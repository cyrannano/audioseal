import os
import random
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import threading
from time import sleep

# Define the directory paths
audio_files_path = 'cremad'
recorded_files_path = 'cremad-recorded'

# Ensure the recorded files directory exists
os.makedirs(recorded_files_path, exist_ok=True)

# Parameters
fs = 16000  # Sample rate
subset_size = 100  # Number of files in the random subset

def play_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    audio.export("temp_audio.wav", format="wav")
    os.system("ffplay -v 0 -nodisp -autoexit temp_audio.wav")

def record_audio(file_name, duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(file_name, fs, recording)  # Save as WAV file
    print(f"Recording saved as {file_name}")

def play_and_record(audio_file, recorded_file):
    audio = AudioSegment.from_wav(audio_file)
    duration = len(audio) / 1000  # duration in seconds

    # Create threads for playing and recording
    play_thread = threading.Thread(target=play_audio, args=(audio_file,))
    record_thread = threading.Thread(target=record_audio, args=(recorded_file, duration, fs))

    # Start both threads
    play_thread.start()
    record_thread.start()

    # Wait for both threads to finish
    play_thread.join()
    record_thread.join()

def main(random_subset=False, subset_size=10):
    audio_files = [f for f in os.listdir(audio_files_path) if f.endswith('.wav')]

    if random_subset:
        audio_files = random.sample(audio_files, subset_size)

    for audio_file in audio_files:
        file_path = os.path.join(audio_files_path, audio_file)
        output_file = os.path.join(recorded_files_path, f"{audio_file}")

        print(f"Processing {file_path}")
        play_and_record(file_path, output_file)

if __name__ == "__main__":
    # Set random_subset to True and specify the subset_size if you want to process a random subset of files
    sleep(10)
    main(random_subset=True, subset_size=100)
