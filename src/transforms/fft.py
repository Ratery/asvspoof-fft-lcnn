import torch
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

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_frames = target_frames

        window = window_fn(win_length)
        self.register_buffer("window", window)

    def forward(self, audio):
        spec = torch.stft(
            input=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        log_spec = torch.log(spec.abs().pow(2) + 1e-9)
        log_spec = log_spec.unsqueeze(1)
        return self._to_fixed_frames(log_spec, self.target_frames)

    def _to_fixed_frames(self, spec, target_frames):
        if spec.shape[-1] < target_frames:
            pad_len = target_frames - spec.shape[-1]
            return nn.functional.pad(spec, (0, pad_len), value=0.0)
        else:
            return spec[..., :target_frames]
