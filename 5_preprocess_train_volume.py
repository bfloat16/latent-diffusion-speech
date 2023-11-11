import os
import numpy as np
import random
import librosa
import torch
import argparse
from logger import utils
from tqdm import tqdm
from tools.tools import Volume_Extractor

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='configs/config.yaml')
    parser.add_argument(
        "-i",
        "--input_txt",
        type=str,
        default=None)
    return parser.parse_args(args=args, namespace=namespace)

def preprocess(path, volume_extractor, sample_rate, device='cuda', input_txt=None):
    path_srcdir = os.path.join(path, 'audio')
    path_volumedir = os.path.join(path, 'volume')
    path_augvoldir = os.path.join(path, 'aug_vol')

    def process(file):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
            
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        volume = volume_extractor.extract(audio, sr=sample_rate)

        max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
        max_shift = min(1, np.log10(1 / max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)

        aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift), sr=sample_rate)

        os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
        np.save(path_volumefile, volume)

        os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
        np.save(path_augvolfile, aug_vol)

    def read_txt_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    filelist = read_txt_file(input_txt)

    for file in tqdm(filelist, total=len(filelist)):
        process(file)

if __name__ == '__main__':
    cmd = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions

    volume_extractor = Volume_Extractor(
        hop_size=512,
        block_size=args.data.block_size,
        model_sampling_rate=args.data.sampling_rate
    )

    preprocess(args.data.train_path, volume_extractor, sample_rate, device=device, input_txt=cmd.input_txt)
