import numpy as np
import soundfile
import torch
import wavmark

class WavMarkWatermarker():
    def __init__(self, device, secret_message = None):
        self.device = device
        self.model = wavmark.load_model().to(device)
        if secret_message is None:
            self.secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
            print(f"Secret message: {secret_message}")
        else:
            self.secret_message = torch.tensor(secret_message, dtype=torch.int32)
    
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        try:
            while tensor.dim() > 1:
                tensor = tensor.squeeze(0)
            audio = tensor.to(self.device)
            # make sure audio is mono

            watermarked_audio, _ = wavmark.encode_watermark(self.model, audio.cpu().numpy(), self.secret_message.squeeze(0))
            return torch.Tensor(watermarked_audio).unsqueeze(0)
        except Exception as e:
            print(f"Error while watermarking audio: {e}")
            return None
        
    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,
        sample_rate: int,
        message_threshold: float = 0.50
    ) -> float:
        try:
            while tensor.dim() > 1:
                    tensor = tensor.squeeze(0)
            decoded_message, _ = wavmark.decode_watermark(self.model, tensor.cpu())
            ber = (decoded_message != self.secret_message.numpy()).astype(np.float32).mean()
            return 1 - ber
        except Exception as e:
            print(f"Error while detecting watermark: {e}")
            return None
