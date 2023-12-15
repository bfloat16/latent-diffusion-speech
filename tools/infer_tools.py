import numpy as np
import torch
import torch.nn.functional
from torchaudio.transforms import Resample
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
        self.spk_emb_dict = None
        self.resample_dict_16000 = {}
        self.units_indexer = None
        self.naive_model_path = None
        self.naive_model = None
        self.naive_model_args = None
        self.use_combo_model = False

    def load_model(self, model_path, f0_model=None, f0_min=None, f0_max=None):
            
        self.model_path = model_path
        self.model, self.vocoder, self.args = load_model_vocoder(model_path, device=self.device)

        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            device=self.device,
            units_forced_mode=self.args.data.units_forced_mode
        )

        self.volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )

        self.load_f0_extractor(f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)

    def flush_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
        if (f0_model != self.f0_model) and (f0_model is not None):
            self.load_f0_extractor(f0_model)

    def load_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
        self.f0_model = f0_model if (f0_model is not None) else self.args.data.f0_extractor
        self.f0_min = f0_min if (f0_min is not None) else self.args.data.f0_min
        self.f0_max = f0_max if (f0_max is not None) else self.args.data.f0_max
        self.f0_model = f0_model
        self.f0_extractor = F0_Extractor(
            f0_extractor=self.f0_model,
            sample_rate=44100,
            hop_size=512,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )

    @torch.no_grad()
    def encode_units(self, audio, sr=44100, padding_mask=None):
        assert self.units_encoder is not None
        hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
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
    def extract_mel(self, audio, sr=44100):
        assert sr == 441000
        mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        return mel

    @torch.no_grad()
    def encode_spk(self, audio, sr=44100):
        assert self.speaker_encoder is not None
        return self.speaker_encoder(audio=audio, sample_rate=sr)

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:  # for realtime speedup
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return torch.nn.functional.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
    def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0,
                 gt_spec=None, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                 spk_emb=None):

        if self.args.model.k_step_max is not None:
            if k_step is None:
                raise ValueError("k_step must not None when Shallow Diffusion Model inferring")
            if k_step > int(self.args.model.k_step_max):
                raise ValueError(f"k_step must <= k_step_max of Shallow Diffusion Model")
            if gt_spec is None:
                raise ValueError("gt_spec must not None when Shallow Diffusion Model inferring, gt_spec can from "
                                 "input mel or output of naive model")

        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)

        # spk_id
        spk_emb_dict = None
        if self.args.model.use_speaker_encoder:  # with speaker encoder
            spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
        # without speaker encoder
        else:
            spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)

        return self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)

    @torch.no_grad()  # 比__call__多了声码器代码，输出波形
    def infer(self, units, f0, volume, gt_spec=None, spk_id=1, spk_mix_dict=None, aug_shift=0,
              infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
              spk_emb=None):
        if k_step is not None:
            if self.naive_model is not None:
                gt_spec = self.naive_model_call(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                                aug_shift=aug_shift, spk_emb=spk_emb)
            assert gt_spec is not None
            if self.naive_model is None:
                if input(" [WARN] You are attempting shallow diffusion "
                         "on the mel of the input source,"
                         " Please enter 'gt_mel' to continue") != 'gt_mel':
                    raise ValueError("Please understand what you're doing")
            k_step = int(k_step)
            gt_spec = gt_spec
        else:
            gt_spec = None
        
        out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                                gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                use_tqdm=use_tqdm, spk_emb=spk_emb)
        
        if self.f0_extractor.f0_extractor == "fcpe" and f0 == None:
            f0 = self.f0_extractor.extract(None, device = out_mel.device, mel = out_mel)
            f0 = torch.tensor(f0[None,:,None],device=out_mel.device)

        return self.mel2wav(out_mel, f0)

    @torch.no_grad()  # 为实时浅扩散优化的推理代码，可以切除pad省算力
    def infer_for_realtime(self, units, f0, volume, audio_t=None, spk_id=1, spk_mix_dict=None, aug_shift=0,
                           infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                           spk_emb=None, silence_front=0, diff_jump_silence_front=False):

        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)

        if diff_jump_silence_front:
            if audio_t is not None:
                audio_t = audio_t[:, start_frame * self.vocoder.vocoder_hop_size:]
            f0 = f0[:, start_frame:, :]
            units = units[:, start_frame:, :]
            volume = volume[:, start_frame:, :]

        if k_step is not None:
            assert audio_t is not None
            k_step = int(k_step)
            gt_spec = self.vocoder.extract(audio_t, self.args.data.sampling_rate)
            # 如果缺帧再开这行gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
        else:
            gt_spec = None

        out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                                gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                use_tqdm=use_tqdm, spk_emb=spk_emb)

        if diff_jump_silence_front:
            out_wav = self.mel2wav(out_mel, f0)
        else:
            out_wav = self.mel2wav(out_mel, f0, start_frame=start_frame)
        return out_wav

    @torch.no_grad()  # 不切片从音频推理代码
    def infer_from_audio(self, audio, sr=44100, key=0, spk_id=1, spk_mix_dict=None, aug_shift=0,
                         infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                         spk_emb=None, threhold=-60, index_ratio=0):
        units = self.encode_units(audio, sr)
        if index_ratio > 0:
            units = self.units_indexer(units_t=units, spk_id=spk_id, ratio=index_ratio)
        f0 = self.extract_f0(audio, key=key, sr=sr)
        volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))
        if k_step is not None:
            assert 0 < int(k_step) <= 1000
            k_step = int(k_step)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            gt_spec = self.vocoder.extract(audio_t, sr)
            gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
        else:
            gt_spec = None
        output = self.infer(units, f0, volume, gt_spec=gt_spec, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                            aug_shift=aug_shift, infer_speedup=infer_speedup, method=method, k_step=k_step,
                            use_tqdm=use_tqdm, spk_emb=spk_emb)
        output *= mask
        return output.squeeze().cpu().numpy(), self.args.data.sampling_rate

    @torch.no_grad()  # 切片从音频推理代码
    def infer_from_long_audio(self, audio, sr=44100, key=0, spk_id=1, spk_mix_dict=None, aug_shift=0,
                              infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                              spk_emb=None,
                              threhold=-60, threhold_for_split=-40, min_len=5000, index_ratio=0):

        hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
        segments = split(audio, sr, hop_size, db_thresh=threhold_for_split, min_len=min_len)

        f0 = self.extract_f0(audio, key=key, sr=sr)
        volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))

        if k_step is not None:
            assert 0 < int(k_step) <= 1000
            k_step = int(k_step)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            gt_spec = self.vocoder.extract(audio_t, sr)
            gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
        else:
            gt_spec = None

        result = np.zeros(0)
        current_length = 0
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
            seg_units = self.units_encoder.encode(seg_input, sr, hop_size)
            if index_ratio > 0:
                seg_units = self.units_indexer(units_t=seg_units, spk_id=spk_id, ratio=index_ratio)
            seg_f0 = f0[:, start_frame: start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame: start_frame + seg_units.size(1), :]
            if gt_spec is not None:
                seg_gt_spec = gt_spec[:, start_frame: start_frame + seg_units.size(1), :]
            else:
                seg_gt_spec = None
            seg_output = self.infer(seg_units, seg_f0, seg_volume, gt_spec=seg_gt_spec, spk_id=spk_id,
                                    spk_mix_dict=spk_mix_dict,
                                    aug_shift=aug_shift, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                    use_tqdm=use_tqdm, spk_emb=spk_emb)
            _left = start_frame * self.args.data.block_size
            _right = (start_frame + seg_units.size(1)) * self.args.data.block_size
            seg_output *= mask[:, _left:_right]
            seg_output = seg_output.squeeze().cpu().numpy()
            silent_length = round(start_frame * self.args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

        return result, self.args.data.sampling_rate

    @torch.no_grad()  # 为实时优化的推理代码，可以切除pad省算力
    def infer_from_audio_for_realtime(self, audio, sr, key, spk_id=1, spk_mix_dict=None, aug_shift=0,
                                      infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                                      spk_emb=None, silence_front=0, diff_jump_silence_front=False, threhold=-60,
                                      index_ratio=0, use_hubert_mask=False):

        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        key_str = str(sr)
        if key_str not in self.resample_dict_16000:
            self.resample_dict_16000[key_str] = Resample(sr, 16000, lowpass_filter_width=128).to(self.device)
        if int(sr) != 16000:
            audio_t_16k = self.resample_dict_16000[key_str](audio_t)
        else:
            audio_t_16k = audio_t

        volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))
        if use_hubert_mask:
            mask16k = mask.clone().unsqueeze(0).unsqueeze(0)
            mask16k = torch.nn.functional.interpolate(mask16k, size=tuple(audio_t_16k.shape), mode='nearest')
            mask16k = ~(mask16k.squeeze(0).squeeze(0).bool())
        else:
            mask16k = None
        units = self.encode_units(audio_t_16k, sr=16000, padding_mask=mask16k)
        if index_ratio > 0:
            units = self.units_indexer(units_t=units, spk_id=spk_id, ratio=index_ratio)
        f0 = self.extract_f0(audio, key=key, sr=sr, silence_front=silence_front)

        if diff_jump_silence_front:
            audio_t = audio_t[:, start_frame * self.vocoder.vocoder_hop_size:]
            f0 = f0[:, start_frame:, :]
            units = units[:, start_frame:, :]
            volume = volume[:, start_frame:, :]

        if k_step is not None:
            k_step = int(k_step)
            if (k_step >= 1000) or (k_step <= 0):
                k_step = 300
                print(f" [WARN] k_step must < 1000 and > 0, now set to {k_step}")
            if self.args.model.k_step_max is not None:
                k_step_max = int(self.args.model.k_step_max)
                if k_step > k_step_max:
                    print(f" [WARN] k_step must <= k_step_max={k_step_max}, not k_step set to{k_step_max}.")
                    k_step = k_step_max
            if int(k_step/infer_speedup) < 3:
                infer_speedup = int(k_step/4)
                print(f" [WARN] diffusion step must > 4 (3 when qndm), not set to{infer_speedup}")
            if self.naive_model is not None:
                gt_spec = self.naive_model_call(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                                aug_shift=aug_shift, spk_emb=spk_emb)
            else:
                gt_spec = self.vocoder.extract(audio_t, self.args.data.sampling_rate)
                gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)

        else:
            gt_spec = None

        out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                                gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                use_tqdm=use_tqdm, spk_emb=spk_emb)

        if diff_jump_silence_front:
            out_wav = self.mel2wav(out_mel, f0)
        else:
            out_wav = self.mel2wav(out_mel, f0, start_frame=start_frame)
            out_wav *= mask
        return out_wav.squeeze(), self.args.data.sampling_rate
