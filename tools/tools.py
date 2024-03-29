import numpy as np
import torch
import librosa
import torch.nn as nn
from fairseq import checkpoint_utils
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from torchaudio.transforms import Resample
from torch.optim.lr_scheduler import StepLR
from encoder.whisper.audio import log_mel_spectrogram
from encoder.whisper.model import ModelDimensions, Whisper

class Volume_Extractor:
    def __init__(self, hop_size=512, block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.hop_size = hop_size
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, sr=None):
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume

    def get_mask_from_volume(self, volume, threhold=-60.0,device='cpu'):
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.block_size).squeeze(-1)
        return mask

class Units_Encoder:
    def __init__(self, encoder, encoder_sample_rate=16000, encoder_hop_size=320, device=None, units_forced_mode='nearest'):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.encoder = encoder

        if units_forced_mode is None:
            units_forced_mode = 'left'
        self.units_forced_mode = units_forced_mode
        
        is_loaded_encoder = False
        if encoder == 'whisper_large_v3':
            self.model = WhisperLargeV3(device=device)
            is_loaded_encoder = True
        if encoder == 'w2v-bert':
            self.model = Wav2Vec2Bert(device=device)
            is_loaded_encoder = True
        if encoder == 'xlsr_53_56k':
            self.model = Audio2xlsr_53_56k(device=device)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f"[x] Unknown units encoder: {encoder}")

        if self.units_forced_mode == 'rfa512to441':
            encoder_sample_rate = encoder_sample_rate * 441 / 512
        if self.units_forced_mode == 'rfa441to512':
            encoder_sample_rate = encoder_sample_rate * 512 / 441

        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

    def encode(self, audio, sample_rate, padding_mask=None):
        if self.units_forced_mode not in ('rfa441to512', 'rfa512to441'):
            if sample_rate == self.encoder_sample_rate:
                audio_res = audio
            else:
                key_str = str(sample_rate)
                if key_str not in self.resample_kernel:
                    self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate).to(self.device)
                audio_res = self.resample_kernel[key_str](audio)
        else:
            if isinstance(audio, np.ndarray):
                _audio = audio
            else:
                _audio = audio.cpu().numpy()    
            audio_res = librosa.resample(_audio, orig_sr=sample_rate, target_sr=self.encoder_sample_rate)
            audio_res = torch.from_numpy(audio_res).to(self.device)

        if self.encoder == 'w2v-bert' and isinstance(audio_res, torch.Tensor):
            audio_res = audio_res.cpu().numpy()

        if isinstance(audio_res, torch.Tensor) and audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res, padding_mask=padding_mask)

        if units.shape[0] == 1:
            units = units.squeeze(0)

        return units

class WhisperLargeV3(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print('whisper_large_v3')
        checkpoint = torch.load('pretrain/large-v3_encoder.pt', map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, audio, padding_mask=None):
        audio = audio.view(1,-1)
        mel = log_mel_spectrogram(audio).to(self.device)
        with torch.no_grad():
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            units = self.model.encoder(mel).squeeze().data.cpu().float()
            return units
        
class Wav2Vec2Bert:
    def __init__(self, device='cpu'):
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", cache_dir="pretrain")
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", cache_dir="pretrain")
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, audio, padding_mask=None):  # B, T
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        for k, input in inputs.items():
            inputs[k] = input.to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
class Audio2xlsr_53_56k():
    def __init__(self, path='pretrain/xlsr_53_56k.pt', device='cpu'):
        self.device = device
        print('xlsr_53_56k')
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device)
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits["x"]
            return units
        
class StepLRWithWarmUp(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, warm_up_steps=1000, start_lr = 1e-6, verbose=False):
        self.warm_up_steps = warm_up_steps
        self.start_lr = start_lr
        super().__init__(optimizer,step_size, gamma, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:
            return [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / self.warm_up_steps
                    for base_lr in self.base_lrs]
        else:
            return super().get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warm_up_steps:
            return [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / self.warm_up_steps
                    for base_lr in self.base_lrs]
        else:
            return super()._get_closed_form_lr()

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def units_forced_alignment(units, audio=None, sample_rate=None, hop_size=None, n_frames=None, scale_factor=None, units_forced_mode='nearest', device='cpu'):
    assert (audio is not None and sample_rate is not None and hop_size is not None) or n_frames is not None or scale_factor is not None
    n_frames = int(audio.size(-1) // hop_size + 1) if n_frames is None else n_frames
    unit_is_tensor = True
    units_dim = len(units.shape)
    if isinstance(units, np.ndarray):
        units = torch.from_numpy(units) 
        unit_is_tensor = False
    if units_dim == 2:
        units = units.unsqueeze(0)

    if units_forced_mode == 'left':
        assert scale_factor is not None
        index = torch.clamp(torch.round(scale_factor * torch.arange(n_frames).to(device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))

    elif units_forced_mode in ('rfa441to512', 'rfa512to441'):
        units = units.transpose(1, 2)
        units_aligned = torch.nn.functional.interpolate(units, size=n_frames, scale_factor=scale_factor, mode='nearest')
        units_aligned = units_aligned.transpose(-1, -2)

    else:
        units = units.transpose(-1, -2)
        units_aligned = torch.nn.functional.interpolate(units, size=n_frames, scale_factor=scale_factor, mode=units_forced_mode)
        units_aligned = units_aligned.transpose(-1, -2)
    
    if not unit_is_tensor:
        units_aligned = units_aligned.numpy()
    if units_dim == 2:
        units_aligned = units_aligned.squeeze(0)
    return units_aligned

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)

def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def get_encdoer_out_channels(encoder):
    if encoder == 'whisper_large_v3':
        return 1280
    elif encoder == 'contentvec768l12':
        return 768
    elif encoder == 'xlsr_53_56k':
        return 1024
    raise ValueError(f"[x] Unknown encoder: {encoder}")