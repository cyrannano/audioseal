import numpy as np
import torch
from typing import Tuple

class Watermarker():
    def generate_watermark_audio() -> torch.Tensor:
        """
        Generate watermark for audio signal
        Returns:
            torch.Tensor: Watermarked audio signal
        """
        return NotImplementedError

    def detect_watermark_audio() -> float:
        """
        Detect watermark in audio signal
        Returns:
            float: probability of watermark being present
        """
        return NotImplementedError

    def detect_watermark_audio_sample_level() -> torch.Tensor:
        """
        Detect watermark in audio signal at sample level
        Returns:
            float: probability of watermark being present
        """
        return NotImplementedError