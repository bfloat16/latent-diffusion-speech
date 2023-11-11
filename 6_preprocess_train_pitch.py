import os
import numpy as np
import argparse
from logger import utils
from tqdm import tqdm
from logger.utils import traverse_dir

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default='configs/config.yaml',
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

def preprocess(path, extensions=['wav']):
    path_srcdir = os.path.join(path, 'audio')
    filelist = traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)

    pitch_aug_dict = {}

    def process(file):
        keyshift = 0
        pitch_aug_dict[file] = keyshift

    for file in tqdm(filelist, total=len(filelist)):
        process(file)

    path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
    np.save(path_pitchaugdict, pitch_aug_dict)

if __name__ == '__main__':
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    use_pitch_aug = False
    extensions = args.data.extensions

    preprocess(args.data.train_path, extensions=extensions)

