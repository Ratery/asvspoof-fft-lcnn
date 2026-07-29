import torch
import torchaudio
from torch import nn


class FFTLogPowerSpectrum(nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        window_fn=torch.blackman_window,
        target_frames: int = 600,
    ):
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            power=2,
            normalized=False,
        )
        self.target_frames = target_frames

    def forward(self, audio):
        power_spec = self.spec(audio)
        log_spec = torch.log(power_spec + 1e-9)
        return self._to_fixed_frames(log_spec, self.target_frames)

    def _to_fixed_frames(self, spec, target_frames):
        if spec.shape[-1] < target_frames:
            pad_len = target_frames - spec.shape[-1]
            return nn.functional.pad(spec, (0, pad_len), value=0.0)
        else:
            return spec[:, :target_frames]
