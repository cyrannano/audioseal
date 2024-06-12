import numpy as np
import torch
from audioseal import AudioSeal
from .Watermarker import Watermarker
import typing as tp

class AudioSealWatermarker(Watermarker):
    def __init__(self, device, secret_message = None, generator="audioseal_wm_16bits", detector="audioseal_detector_16bits"):
        self.device = device
        self.model = AudioSeal.load_generator(generator)
        self.detector = AudioSeal.load_detector(detector)
        self.model = self.model.to(device)
        self.detector = self.detector.to(device)
        if secret_message is None:
            self.secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
            self.secret_message = self.secret_message.to(device)
            print(f"Secret message: {self.secret_message}")
        else:
            self.secret_message = torch.tensor(secret_message, dtype=torch.int32)
            self.secret_message = self.secret_message.to(device)

        
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,
        sample_rate: int
    ) -> tp.Optional[torch.Tensor]:
        try:
            audios = tensor.unsqueeze(0).to(self.device)
            watermarked_audio = self.model(audios, sample_rate=sample_rate, message=self.secret_message.to(self.device), alpha=1)
            return watermarked_audio

        except Exception as e:
            print(f"Error while watermarking audio: {e}")
            return None

    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,
        sample_rate: int,
        message_threshold: float = 0.50
    ) -> tp.Optional[float]:
        try:
            # In our analysis we are not concerned with the hidden/embedded message as of now
            result, _ = self.detector.detect_watermark(tensor, sample_rate=sample_rate, message_threshold=message_threshold)
            return float(result)
        except Exception as e:
            print(f"Error while detecting watermark: {e}")
            return None
        
    def detect_watermark_audio_sample_level(
        self,
        tensor: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        try:
            result = self.detector.forward(tensor, sample_rate=sample_rate)
            return result
        except Exception as e:
            print(f"Error while detecting watermark: {e}")
            return None