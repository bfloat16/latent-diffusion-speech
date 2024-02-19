import os
import numpy as np
import librosa
import argparse
import torch
import random
from glob import glob
from tools import utils
from diffusion.vocoder import Vocoder
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def preprocess(path, sample_rate, type, ckpt, device='cuda'):
    mel_extractor = Vocoder(type, ckpt, device=device)

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_melfile = file.replace('audio', 'mel', 1)
            path_augmelfile = file.replace('audio', 'aug_mel', 1)

            audio, _ = librosa.load(file, sr=sample_rate)
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
    parser.add_argument("-n", "--num_processes", type=int, default=4)
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_processes = cmd.num_processes
    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    type = args.common.vocoder.type
    ckpt = args.common.vocoder.ckpt
    only_mean=args["vocoder"]["only_mean"]

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filelist) / num_processes)
            end = int((i + 1) * len(filelist) / num_processes)
            file_chunk = filelist[start:end]
            tasks.append(executor.submit(preprocess, file_chunk, sample_rate, type, ckpt, device='cuda', use_pitch_aug=False))
        for task in tasks:
            task.result()