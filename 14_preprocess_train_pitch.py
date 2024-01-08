import os
import numpy as np
import argparse
from logger import utils
from logger.utils import traverse_dir

def preprocess(path, extensions=['wav']):
    path_srcdir = os.path.join(path, 'audio')
    filelist = traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)

    pitch_aug_dict = {}

    for file in filelist:
        keyshift = 0
        pitch_aug_dict[file] = keyshift

    path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
    np.save(path_pitchaugdict, pitch_aug_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml', help="path to the config file")
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    extensions = args.data.extensions

    preprocess(args.data.train_path, extensions=extensions)