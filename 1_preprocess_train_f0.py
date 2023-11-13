import os
import numpy as np
import librosa
import argparse
from glob import glob
import torch.multiprocessing as mp
from logger import utils
from tools.tools import F0_Extractor
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

def preprocess(rank, path, f0_extractor, sample_rate, num_workers):
    path = path[rank::num_workers]

    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(path))
        for file in path:
            path_f0file = file.replace('audio', 'f0', 1)

            audio, _ = librosa.load(file, sr=sample_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            f0 = f0_extractor.extract(audio, uv_interp=False, sr=sample_rate)

            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
                os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
                np.save(path_f0file, f0)
            else:
                print('\n[Error] F0 extraction failed: ' + file)
                os.remove(file)
                os.remove(file.replace('.wav', '.lab'))
            rich_progress.update(rank, advance=1)

def main(train_path, f0_extractor, sample_rate, num_workers=1):
    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    manager = mp.Manager()
    data_q = manager.Queue()

    def put_f0_extractor(queue, mel_extractor):
        queue.put(mel_extractor)

    receiver = mp.Process(target=put_f0_extractor, args=(data_q, f0_extractor))
    receiver.start()
    mp.spawn(preprocess, args=(filelist, data_q.get(), sample_rate, num_workers), nprocs=num_workers, join=True)
    receiver.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    cmd = parser.parse_args()

    args = utils.load_config(cmd.config)
    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    num_workers = cmd.num_workers

    f0_extractor = F0_Extractor(f0_extractor=args.data.f0_extractor, sample_rate=44100, hop_size=512, f0_min=args.data.f0_min, f0_max=args.data.f0_max, block_size=args.data.block_size, model_sampling_rate=args.data.sampling_rate)

    main(train_path, f0_extractor, sample_rate, num_workers)
