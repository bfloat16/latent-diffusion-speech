import os
import numpy as np
import librosa
import torch
import torch.multiprocessing as mp
import argparse
from logger import utils
from glob import glob
from tools.tools import Units_Encoder
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import warnings
warnings.filterwarnings("ignore")

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

def preprocess(rank, path, sample_rate, hop_size, num_workers, encoder, encoder_ckpt, encoder_sample_rate, encoder_hop_size, units_forced_mode, device='cuda'):
    if encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(encoder, encoder_ckpt, encoder_sample_rate, encoder_hop_size, cnhubertsoft_gate=cnhubertsoft_gate, device=device, units_forced_mode=units_forced_mode)
    path = path[rank::num_workers]

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_unitsfile = file.replace('audio', 'units', 1)
                
            audio, _ = librosa.load(file, sr=sample_rate)
            audio = librosa.to_mono(audio)

            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)

            units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
            units = units_t.squeeze().to('cpu').numpy()

            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            rich_progress.update(rank, advance=1)

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_workers", type=int, default=3)
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)
    
    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    num_workers = cmd.num_workers
    encoder = args.data.encoder
    encoder_ckpt = args.data.encoder_ckpt
    encoder_sample_rate = args.data.encoder_sample_rate
    encoder_hop_size = args.data.encoder_hop_size
    units_forced_mode = args.data.units_forced_mode

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    mp.spawn(preprocess, args=(filelist, sample_rate, hop_size, num_workers, encoder, encoder_ckpt, encoder_sample_rate, encoder_hop_size, units_forced_mode), nprocs=num_workers, join=True)

if __name__ == '__main__':
    main()