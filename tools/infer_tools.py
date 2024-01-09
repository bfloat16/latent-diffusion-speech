import numpy as np
import torch
import torch.nn.functional
from tqdm import tqdm
from diffusion.unit2mel import load_model_vocoder
from tools.slicer import split
from tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder, cross_fade

class DiffusionSVC:
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = None
        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.f0_extractor = None
        self.f0_model = None
        self.f0_min = None
        self.f0_max = None
        self.volume_extractor = None
        self.speaker_encoder = None
        self.resample_dict_16000 = {}
        self.naive_model_path = None
        self.naive_model = None
        self.naive_model_args = None
        self.use_combo_model = False

    def load_model(self, model_path, f0_min=None, f0_max=None):
        self.model_path = model_path
        self.model, self.vocoder, self.args = load_model_vocoder(model_path, device=self.device)

        self.units_encoder = Units_Encoder(
            self.args['data']['encoder'],
            self.args['data']['encoder_sample_rate'],
            self.args['data']['encoder_hop_size'],
            device=self.device,
        )

        self.volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=self.args['data']['block_size'],
            model_sampling_rate=self.args['data']['sampling_rate']
        )

        self.load_f0_extractor(f0_min=f0_min, f0_max=f0_max)

    def load_f0_extractor(self, f0_min=None, f0_max=None):
        self.f0_min = f0_min if (f0_min is not None) else self.args['data']['f0_min']
        self.f0_max = f0_max if (f0_max is not None) else self.args['data']['f0_max']
        self.f0_extractor = F0_Extractor(
            sample_rate=44100,
            hop_size=512,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            block_size=self.args['data']['block_size'],
            model_sampling_rate=self.args['data']['sampling_rate']
        )

    @torch.no_grad()
    def encode_units(self, audio, sr=44100, padding_mask=None):
        assert self.units_encoder is not None
        hop_size = self.args['data']['block_size'] * sr / self.args['data']['sampling_rate']
        return self.units_encoder.encode(audio, sr, hop_size, padding_mask=padding_mask)

    @torch.no_grad()
    def extract_f0(self, audio, key=0, sr=44100, silence_front=0):
        assert self.f0_extractor is not None
        f0 = self.f0_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front, sr=sr)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(key) / 12)
        return f0

    @torch.no_grad()
    def extract_volume_and_mask(self, audio, sr=44100, threhold=-60.0):
        assert self.volume_extractor is not None
        volume = self.volume_extractor.extract(audio, sr)
        mask = self.volume_extractor.get_mask_from_volume(volume, threhold=threhold, device=self.device)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        return volume, mask

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return torch.nn.functional.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
    def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0, gt_spec=None, infer_speedup=10, method='unipc', use_tqdm=True):
        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)
        spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)

        return self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, use_tqdm=use_tqdm)

    @torch.no_grad()  # 比__call__多了声码器代码，输出波形
    def infer(self, units, f0, volume, gt_spec=None, spk_id=1, spk_mix_dict=None, aug_shift=0, infer_speedup=10, method='unipc', use_tqdm=True):
        gt_spec = None
        out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, use_tqdm=use_tqdm)
        
        if self.f0_extractor.f0_extractor == "fcpe" and f0 == None:
            f0 = self.f0_extractor.extract(None, device = out_mel.device, mel = out_mel)
            f0 = torch.tensor(f0[None,:,None],device=out_mel.device)

        return self.mel2wav(out_mel, f0)

    @torch.no_grad()  # 切片从音频推理代码
    def infer_from_long_audio(self, audio, sr=44100, key=0, spk_id=1, spk_mix_dict=None, aug_shift=0, infer_speedup=10, method='unipc', use_tqdm=True, threhold=-60, threhold_for_split=-40, min_len=5000):
        hop_size = self.args['data']['block_size'] * sr / self.args['data']['sampling_rate']
        segments = split(audio, sr, hop_size, db_thresh=threhold_for_split, min_len=min_len)

        f0 = self.extract_f0(audio, key=key, sr=sr)
        volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))
        gt_spec = None

        result = np.zeros(0)
        current_length = 0
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
            seg_units = self.units_encoder.encode(seg_input, sr, hop_size)
            seg_f0 = f0[:, start_frame: start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame: start_frame + seg_units.size(1), :]
            if gt_spec is not None:
                seg_gt_spec = gt_spec[:, start_frame: start_frame + seg_units.size(1), :]
            else:
                seg_gt_spec = None
            seg_output = self.infer(seg_units, seg_f0, seg_volume, gt_spec=seg_gt_spec, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer_speedup=infer_speedup, method=method, use_tqdm=use_tqdm)
            _left = start_frame * self.args['data']['block_size']
            _right = (start_frame + seg_units.size(1)) * self.args['data']['block_size']
            seg_output *= mask[:, _left:_right]
            seg_output = seg_output.squeeze().cpu().numpy()
            silent_length = round(start_frame * self.args['data']['block_size']) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

        return result, self.args['data']['sampling_rate']