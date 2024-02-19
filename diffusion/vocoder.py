import torch
from torchaudio.transforms import Resample
from encoder.hifi_vaegan.hifi_vaegan import Hifi_VAEGAN

class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.vocoder_type = vocoder_type
        if vocoder_type == 'hifi-vaegan':
            self.vocoder = Hifi_VAEGAN(vocoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate, keyshift=0, **kwargs):
        if sample_rate == self.vocoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.vocoder.extract(audio_res, keyshift=keyshift, **kwargs)  # B, n_frames, bins
        return mel

    def infer(self, mel):
        return self.vocoder(mel)