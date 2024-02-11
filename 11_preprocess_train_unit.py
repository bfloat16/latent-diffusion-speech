import os
import numpy as np
import librosa
import argparse
import torch
from glob import glob
from tools import utils
from tools.tools import Units_Encoder
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def preprocess(path, sample_rate, hop_size, encoder, encoder_sample_rate, encoder_hop_size, units_forced_mode, device='cuda'):
    units_encoder = Units_Encoder(encoder, encoder_sample_rate, encoder_hop_size, device=device, units_forced_mode=units_forced_mode)
    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_unitsfile = file.replace('audio', 'units', 1)

            audio, _ = librosa.load(file)
            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)

            units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
            units = units_t.squeeze().to('cpu').numpy()

            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            rich_progress.update(rank, advance=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_processes", type=int, default=2)
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    num_processes = cmd.num_processes
    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    encoder = args.data.encoder
    encoder_sample_rate = args.data.encoder_sample_rate
    encoder_hop_size = args.data.encoder_hop_size
    units_forced_mode = args.data.units_forced_mode

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filelist) / num_processes)
            end = int((i + 1) * len(filelist) / num_processes)
            file_chunk = filelist[start:end]
            tasks.append(executor.submit(preprocess, file_chunk, sample_rate, hop_size, encoder, encoder_sample_rate, encoder_hop_size, units_forced_mode))
        for task in tasks:
            task.result()