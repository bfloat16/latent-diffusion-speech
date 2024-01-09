import os
import yaml
import torch
import torch.nn as nn
from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .unet1d.unet_1d_condition import UNet1DConditionModel
from tools.tools import get_encdoer_out_channels

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_model_vocoder(
        model_path,
        device='cpu',
        loaded_vocoder=None):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    if loaded_vocoder is None:
        vocoder = Vocoder(args['common']['vocoder']['type'], args['common']['vocoder']['ckpt'], device=device)
    else:
        vocoder = loaded_vocoder

    model = load_svc_model(args=args, vocoder_dimension=vocoder.dimension)

    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args

def load_svc_model(args, vocoder_dimension):
    model = Unit2Mel(
                get_encdoer_out_channels(args['data']['encoder']),
                args['common']['n_spk'],
                args['diffusion']['model']['use_pitch_aug'],
                vocoder_dimension,
                args['diffusion']['model']['n_layers'],
                args['diffusion']['model']['block_out_channels'],
                args['diffusion']['model']['n_heads'],
                args['diffusion']['model']['n_hidden'],
                is_tts = args['diffusion']['model']['is_tts']
                )
    return model

class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=2,
            block_out_channels=(256,384,512,512),
            n_heads=8,
            n_hidden=256,
            is_tts: bool = False
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.is_tts = is_tts
        if not is_tts:
            self.f0_embed = nn.Linear(1, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            if use_pitch_aug:
                self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
            else:
                self.aug_shift_embed = None
        else:
            self.aug_shift_embed = None
            self.f0_embed = None
            self.volume_embed = None

        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)

        self.decoder = GaussianDiffusion(UNet1DConditionModel(
        in_channels=out_dims + n_hidden,
        out_channels=out_dims,
        block_out_channels=block_out_channels,
        norm_num_groups=8,
        cross_attention_dim = block_out_channels,
        attention_head_dim = n_heads,
        only_cross_attention = True,
        layers_per_block = n_layers,
        resnet_time_scale_shift='scale_shift'), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, aug_shift=None, gt_spec=None, infer=True, infer_speedup=10, method='unipc', use_tqdm=False):
        if f0 is None or self.is_tts:
            f0 = 0
        else:
            f0 = self.f0_embed((1 + f0 / 700).log())
        if volume is None or self.is_tts:
            volume = 0
        else:
            volume = self.volume_embed(volume)

        x = self.unit_embed(units) + f0 + volume

        if self.n_spk is not None and self.n_spk > 1:
            x = x + self.spk_embed(spk_id - 1)
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, use_tqdm=use_tqdm)

        return x