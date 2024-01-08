
import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from logger import utils
from logger.utils import traverse_dir
from text.cleaner import text_to_sequence
from text.multi_language_bert import get_bert_token
from diffusion.vocoder import Vocoder
from tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder

def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, text2semantic_mode, sample_rate, hop_size, device='cuda', use_pitch_aug=False, extensions=['wav'], is_tts=False):
    path_srcdir = os.path.join(path, 'audio')
    path_unitsdir = os.path.join(path, 'units')
    path_f0dir = os.path.join(path, 'f0')
    path_volumedir = os.path.join(path, 'volume')
    path_augvoldir = os.path.join(path, 'aug_vol')
    path_meldir = os.path.join(path, 'mel')
    path_augmeldir = os.path.join(path, 'aug_mel')
    path_skipdir = os.path.join(path, 'skip')
    if is_tts:
        path_uttdir = os.path.join(path, 'utt')

    filelist = traverse_dir(path_srcdir, extensions, is_pure=True, is_sort=True, is_ext=True)
    pitch_aug_dict = {}
    utt_text = {}

    for file in filelist:
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        
        f0 = f0_extractor.extract(audio, uv_interp=False, sr=sample_rate)
        uv = f0 == 0

        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)

            volume = volume_extractor.extract(audio, sr=sample_rate)

            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0

            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift=keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift), sr=sample_rate)

            units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
            units = units_t.squeeze().to('cpu').numpy()

            if is_tts:
                speaker = binfile.split('\\')[0]
                previous_speaker = None
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

            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
            np.save(path_melfile, mel)
            os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
            np.save(path_augmelfile, aug_mel)
            os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
            np.save(path_augvolfile, aug_vol)
            os.makedirs(os.path.dirname(path_uttfile), exist_ok=True)
            np.save(path_uttfile, np.array((np.array(phones), np.array(tones), np.array(lang_ids), np.array(word2ph)),dtype=object), allow_pickle=True)

            pitch_aug_dict[file] = keyshift
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)

        path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
        np.save(path_pitchaugdict, pitch_aug_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/config.yaml")
    parser.add_argument("-t", "--tts", action='store_true', default=True)
    cmd = parser.parse_args()

    is_tts = cmd.tts
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions
    text2semantic_mode = args['text2semantic']['model']['mode']
    use_pitch_aug = args['diffusion']['model']['use_pitch_aug']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    f0_extractor = F0_Extractor(sample_rate=44100, hop_size=512, f0_min=args.data.f0_min, f0_max=args.data.f0_max, block_size=args.data.block_size, model_sampling_rate=args.data.sampling_rate)
    volume_extractor = Volume_Extractor(hop_size=512, block_size=args.data.block_size, model_sampling_rate=args.data.sampling_rate)
    mel_extractor = Vocoder(args.common.vocoder.type, args.common.vocoder.ckpt, device=device)
    units_encoder = Units_Encoder(args.data.encoder, args.data.encoder_sample_rate, args.data.encoder_hop_size, units_forced_mode=args.data.units_forced_mode)

    preprocess(args.data.valid_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, text2semantic_mode, sample_rate, hop_size, device=device, use_pitch_aug=use_pitch_aug, extensions=extensions,is_tts=is_tts)