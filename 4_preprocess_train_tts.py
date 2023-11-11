import os
import numpy as np
import argparse
from logger import utils
from tqdm import tqdm
from logger.utils import traverse_dir
from text.cleaner import text_to_sequence

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/config.yaml",
        required=False,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


def preprocess(path, extensions=['wav']):
    path_srcdir = os.path.join(path, 'audio')
    path_uttdir = os.path.join(path, 'utt')
        
    filelist = traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)

    def process(file):
        binfile = file + '.npy'
        path_uttfile = os.path.join(path_srcdir, file)
        path_uttfile = os.path.dirname(path_uttfile)
        path_uttfile = os.path.join(path_uttfile,"utt_txt.txt")
        with open(path_uttfile,"r",encoding="UTF8") as f:
            utt_text = {}
            for f_i in f.readlines():
                k, v = f_i.replace("\n","").split("|")
                utt_text[k] = v
        path_uttfile = os.path.join(path_uttdir, binfile)
        
        file_name = os.path.split(file)[-1]
        text = utt_text[file_name]
        (phones, tones, lang_ids), (norm_text, word2ph) = text_to_sequence(text, "ZH")

        os.makedirs(os.path.dirname(path_uttfile), exist_ok=True)
        np.save(path_uttfile, np.array((np.array(phones), np.array(tones), np.array(lang_ids), np.array(word2ph)),dtype=object), allow_pickle=True)

    for file in tqdm(filelist, total=len(filelist)):
        process(file)

if __name__ == '__main__':
    cmd = parse_args()

    args = utils.load_config(cmd.config)

    preprocess(args.data.train_path)
