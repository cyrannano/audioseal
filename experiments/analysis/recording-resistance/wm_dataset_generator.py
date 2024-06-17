import os
import torch
import torchaudio
from tqdm import tqdm
from audioseal import AudioSeal

# Load the AudioSeal model and detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioSeal.load_generator("./generator_base.pth", nbits=16).to(device)
# detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)

# Generate a secret message
secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32).to(device)
print(f"Secret message: {secret_message}")

# Function to load an audio file from its file path
def load_audio_file(file_path: str):
    try:
        wav, sample_rate = torchaudio.load(file_path)
        return wav, sample_rate
    except Exception as e:
        print(f"Error while loading audio: {e}")
        return None

# Function to generate a watermark for the audio and embed it into a new audio tensor
def generate_watermark_audio(tensor: torch.Tensor, sample_rate: int):
    try:
        audios = tensor.unsqueeze(0).to(device)
        watermarked_audio = model(audios, sample_rate=sample_rate, message=secret_message.to(device), alpha=1)
        return watermarked_audio
    except Exception as e:
        print(f"Error while watermarking audio: {e}")
        return None

# Create output directory if it does not exist
output_dir = "cremad-wm"
os.makedirs(output_dir, exist_ok=True)

# Process all audio files in the cremad directory
input_dir = "cremad"
audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

for audio_file in tqdm(audio_files, desc="Processing audio files"):
    input_file = os.path.join(input_dir, audio_file)
    output_file = os.path.join(output_dir, audio_file)
    
    audio_data = load_audio_file(input_file)
    if audio_data is None:
        continue
    
    audio, sample_rate = audio_data
    
    watermarked_audio = generate_watermark_audio(audio, sample_rate)
    if watermarked_audio is None:
        continue
    
    try:
        watermarked_audio = watermarked_audio.detach()
        torchaudio.save(output_file, watermarked_audio.squeeze(0).cpu(), sample_rate)
    except Exception as e:
        print(f"Error while saving audio: {e}")
        continue

print("Processing completed.")
