import os
import numpy as np
import librosa
import argparse
from logger import utils
from tqdm import tqdm
from tools.tools import F0_Extractor
import warnings
warnings.filterwarnings("ignore")

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

def preprocess(path, f0_extractor, sample_rate, input_txt=None):
    path_srcdir = os.path.join(path, 'audio')
    path_f0dir = os.path.join(path, 'f0')

    def process(file):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_f0file = os.path.join(path_f0dir, binfile)

        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        f0 = f0_extractor.extract(audio, uv_interp=False, sr=sample_rate)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.remove(path_srcfile)
            os.remove(path_srcfile.replace('.wav', '.lab'))

    def read_txt_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    filelist = read_txt_file(input_txt)

    for file in tqdm(filelist, total=len(filelist)):
        process(file)

if __name__ == '__main__':
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    f0_extractor = F0_Extractor(f0_extractor=args.data.f0_extractor, sample_rate=44100,hop_size=512, f0_min=args.data.f0_min, f0_max=args.data.f0_max, block_size=args.data.block_size, model_sampling_rate=args.data.sampling_rate)

    preprocess(args.data.train_path, f0_extractor, sample_rate, input_txt=cmd.input_txt)
