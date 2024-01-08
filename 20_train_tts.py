import argparse
import torch
from logger import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel
from diffusion.vocoder import Vocoder
import accelerate
import itertools
from tools.tools import StepLRWithWarmUp
from rich.console import Console
from vector_quantize_pytorch import VectorQuantize
from tools.tools import get_encdoer_out_channels

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    args = utils.load_config(cmd.config)
    print('Training Args: ')
    Console().print(args)
    
    vocoder = Vocoder(args['common']['vocoder']['type'], args['common']['vocoder']['ckpt'], device=device)
    
    model = Unit2Mel(
        get_encdoer_out_channels(args['data']['encoder']),
        args['common']['n_spk'],
        args['diffusion']['model']['use_pitch_aug'],
        vocoder.dimension,
        args['diffusion']['model']['n_layers'],
        args['diffusion']['model']['block_out_channels'],
        args['diffusion']['model']['n_heads'],
        args['diffusion']['model']['n_hidden'],
        is_tts=args['diffusion']['model']['is_tts']
        )

    if args['text2semantic']['train']['use_units_quantize']:
        if args['text2semantic']['train']['units_quantize_type'] == "kmeans":
            from quantize.kmeans_codebook import EuclideanCodebook
            from cluster import get_cluster_model
            codebook_weight = get_cluster_model(args['text2semantic']['model']['codebook_path']).__dict__["cluster_centers_"]
            quantizer = EuclideanCodebook(codebook_weight).to(device)
        elif args['text2semantic']['train']['units_quantize_type'] == "vq":
            quantizer = VectorQuantize(
                dim = get_encdoer_out_channels(args['data']['encoder']),
                codebook_size = args['text2semantic']['train']['semantic_kmeans_num'],
                decay = 0.8,
                commitment_weight = 1.
                ).to(device)
        else:
            raise ValueError('[Error] Unknown quantize_type: ' + args['text2semantic']['train']['units_quantize_type'])
        
        optimizer = torch.optim.AdamW(itertools.chain(model.parameters(),quantizer.parameters()))
    else:
        quantizer = None
        optimizer = torch.optim.AdamW(model.parameters())
    
    initial_global_step, model, optimizer = utils.load_model(args['diffusion']['train']['expdir'], model, optimizer, device=args['common']['device'])

    if quantizer is not None and args['text2semantic']['train']['units_quantize_type'] == "vq":
        try:
            _, quantizer, _ = utils.load_model(args['diffusion']['train']['expdir'], quantizer, optimizer, device=args['common']['device'], postfix=f'{initial_global_step}_semantic_codebook')
        except:
            raise ValueError('[Error] Cannot load semantic codebook')
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args['diffusion']['train']['lr']
        param_group['lr'] = args['diffusion']['train']['lr'] * args['diffusion']['train']['gamma'] ** max((initial_global_step - 2) // args['diffusion']['train']['decay_step'], 0)
        param_group['weight_decay'] = args['diffusion']['train']['weight_decay']
        
    scheduler = StepLRWithWarmUp(
        optimizer,
        step_size=args['diffusion']['train']['decay_step'],
        gamma=args['diffusion']['train']['gamma'],
        last_epoch=initial_global_step - 2,
        warm_up_steps=args['diffusion']['train']['warm_up_steps'],
        start_lr=float(args['diffusion']['train']['start_lr']))
    model.to(device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                    
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False,accelerator=accelerator)
    _, model, quantizer, optim, scheduler = accelerator.prepare(loader_train, model, quantizer, optimizer, scheduler)
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, quantizer, accelerator)