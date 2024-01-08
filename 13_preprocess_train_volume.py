import os
import numpy as np
import librosa
import argparse
import torch
import random
from glob import glob
from logger import utils
from tools.tools import Volume_Extractor
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def preprocess(path, sample_rate, block_size, device='cuda'):
    volume_extractor = Volume_Extractor(hop_size=512, block_size=block_size, model_sampling_rate=sample_rate)

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_volumefile = file.replace('audio', 'volume', 1)
            path_augvolfile = file.replace('audio', 'aug_vol', 1)

            audio, _ = librosa.load(file, sr=sample_rate)
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
    parser.add_argument("-n", "--num_processes", type=int, default=1)
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_processes = cmd.num_processes
    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    block_size = args.data.block_size

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filelist) / num_processes)
            end = int((i + 1) * len(filelist) / num_processes)
            file_chunk = filelist[start:end]
            tasks.append(executor.submit(preprocess, file_chunk, sample_rate, block_size, device=device))
        for task in tasks:
            task.result()