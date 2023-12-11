import os
import numpy as np
import torch
import librosa
import argparse
from glob import glob
from logger import utils
from tools.tools import F0_Extractor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")

def preprocess(path, sample_rate, f0_extractor, f0_min, f0_max, block_size, sampling_rate):

    f0_extractor = F0_Extractor(f0_extractor=f0_extractor, sample_rate=44100, hop_size=512, f0_min=f0_min, f0_max=f0_max, block_size=block_size, model_sampling_rate=sampling_rate)

    for file in tqdm(path):
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
            print('[Error] F0 extraction failed: ' + file)
            os.remove(file)
            os.remove(file.replace('.wav', '.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_processes", type=int, default=15)
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    train_path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    num_processes = cmd.num_processes

    f0_extractor=args.data.f0_extractor
    f0_min=args.data.f0_min
    f0_max=args.data.f0_max
    block_size=args.data.block_size
    sampling_rate=args.data.sampling_rate

    filelist = glob(f"{train_path}/audio/**/*.wav", recursive=True)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filelist) / num_processes)
            end = int((i + 1) * len(filelist) / num_processes)
            file_chunk = filelist[start:end]
            tasks.append(executor.submit(preprocess, file_chunk, sample_rate, f0_extractor, f0_min, f0_max, block_size, sampling_rate))
        for task in tasks:
            task.result()
