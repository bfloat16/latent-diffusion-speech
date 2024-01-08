import os
import numpy as np
import argparse
from logger import utils
from logger.utils import traverse_dir
from text.cleaner import text_to_sequence
from text.multi_language_bert import get_bert_token
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def preprocess(path, extensions, text2semantic_mode):
    path_srcdir = os.path.join(path, 'audio')
    path_uttdir = os.path.join(path, 'utt')
    filelist = traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
    utt_text = {}
    previous_speaker = None
    with rich_progress:
        rank = rich_progress.add_task("Preprocess", total=len(filelist))
        for file in filelist:
            binfile = file + '.npy'
            speaker = binfile.split('\\')[0]
            if speaker != previous_speaker:
                path_uttfile = os.path.join(path_srcdir, file)
                path_uttfile = os.path.dirname(path_uttfile)
                path_uttfile = os.path.join(path_uttfile,"utt_text.txt")
                with open(path_uttfile,"r",encoding="UTF8") as f:
                    for f_i in f.readlines():
                        k, v = f_i.replace("\n","").split("|")
                        utt_text[k] = v
                    previous_speaker = speaker
            path_uttfile = os.path.join(path_uttdir, binfile)
            file_name = os.path.split(file)[-1]
            text = utt_text[file_name]
            if text2semantic_mode == "phone":
                (phones, tones, lang_ids), (norm_text, word2ph) = text_to_sequence(text, "JA")
            elif text2semantic_mode == "text":
                tones = lang_ids = word2ph = []
                phones, norm_text = get_bert_token(text)
            os.makedirs(os.path.dirname(path_uttfile), exist_ok=True)
            np.save(path_uttfile, np.array((np.array(phones), np.array(tones), np.array(lang_ids), np.array(word2ph)),dtype=object), allow_pickle=True)
            rich_progress.update(rank, advance=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/config.yaml")
    cmd = parser.parse_args()
    args = utils.load_config(cmd.config)

    train_path = args['data']['train_path']
    extensions = args['data']['extensions']
    text2semantic_mode = args['text2semantic']['model']['mode']

    preprocess(train_path, extensions, text2semantic_mode)