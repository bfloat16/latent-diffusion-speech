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

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/config.yaml",
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    args = utils.load_config(cmd.config)
    Console().print(args)
    
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.block_out_channels,
                args.model.n_heads,
                args.model.n_hidden,
                use_speaker_encoder=args.model.use_speaker_encoder,
                speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
                is_tts=args.model.is_tts
                )

    if args.train.use_units_quantize:
        if args.train.units_quantize_type == "kmeans":
            from quantize.kmeans_codebook import EuclideanCodebook
            from cluster import get_cluster_model
            codebook_weight = get_cluster_model(args.model.text2semantic.codebook_path).__dict__["cluster_centers_"]
            quantizer = EuclideanCodebook(codebook_weight).to(device)
        elif args.train.units_quantize_type == "vq":
            from vector_quantize_pytorch import VectorQuantize
            quantizer = VectorQuantize(
                dim = args.data.encoder_out_channels,
                codebook_size = args.model.text2semantic.semantic_kmeans_num,
                decay = 0.8,             
                commitment_weight = 1. 
            ).to(device)
        else:
            raise ValueError('[x] Unknown quantize_type: ' + args.train.units_quantize_type)
        optimizer = torch.optim.AdamW(itertools.chain(model.parameters(),quantizer.parameters()))
    else:
        quantizer = None
        optimizer = torch.optim.AdamW(model.parameters())
    
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    if quantizer is not None and args.train.units_quantize_type == "vq":
        try:
            _, quantizer, _ = utils.load_model(args.env.expdir, quantizer, optimizer, device=args.device, postfix=f'{initial_global_step}_semantic_codebook')
        except:
            print("[x] No semantic codebook found, use random codebook instead.")
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = StepLRWithWarmUp(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2, warm_up_steps=args.train.warm_up_steps, start_lr=float(args.train.start_lr))
    model.to(device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                    
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False,accelerator=accelerator)
    _, model, quantizer, optim, scheduler = accelerator.prepare(loader_train, model, quantizer, optimizer, scheduler)
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, quantizer, accelerator)
    
