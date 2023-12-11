import os
import numpy as np
import random
import librosa
import torch
import argparse
from logger import utils
import torch.multiprocessing as mp
from tools.tools import Volume_Extractor
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

def preprocess(rank, path, sample_rate, num_workers, block_size, device='cuda'):
    path = path[rank::num_workers]
    volume_extractor = Volume_Extractor(hop_size=512, block_size=block_size, model_sampling_rate=sample_rate)

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_volumefile = file.replace('audio', 'volume', 1)
            path_augvolfile = file.replace('audio', 'aug_vol', 1)
            audio, _ = librosa.load(file, sr=sample_rate)
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
    block_size = args.data.block_size
    num_workers = cmd.num_workers

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    mp.spawn(preprocess, args=(filelist, sample_rate, num_workers, block_size), nprocs=num_workers, join=True)
