import os
import numpy as np
import librosa
import random
import torch
import torch.multiprocessing as mp
import argparse
from logger import utils
from diffusion.vocoder import Vocoder
from glob import glob
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(
    TextColumn("Preprocess:"),
    BarColumn(bar_width=80), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    transient=True
    )

def preprocess(rank, path, sample_rate, num_workers, type, ckpt, device='cuda'):
    path = path[rank::num_workers]
    mel_extractor = Vocoder(type, ckpt, device=device)

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_melfile = file.replace('audio', 'mel', 1)
            path_augmelfile = file.replace('audio', 'aug_mel', 1)

            audio, _ = librosa.load(file, sr=sample_rate)
            audio = librosa.to_mono(audio)

            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift=0)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()

            os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
            np.save(path_melfile, mel)
            os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
            np.save(path_augmelfile, aug_mel)
            rich_progress.update(rank, advance=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    cmd = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = utils.load_config(cmd.config)

    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    num_workers = cmd.num_workers
    type = args.vocoder.type
    ckpt = args.vocoder.ckpt

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    mp.spawn(preprocess, args=(filelist, sample_rate, num_workers, type, ckpt, device), nprocs=num_workers, join=True)

