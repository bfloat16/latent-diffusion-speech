import os
import random
import numpy as np
import librosa
import torch
import random
from torch.utils.data import Dataset
from tools.tools import units_forced_alignment
from logger.utils import traverse_dir
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

progress = Progress(
    TextColumn("Loading: "),
    BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    transient=True
    )

def get_data_loaders(args, whole_audio=False, accelerator=None):
    data_train = AudioDataset(
        args['data']['train_path'],
        waveform_sec=args['data']['duration'],
        hop_size=args['data']['block_size'],
        sample_rate=args['data']['sampling_rate'],
        load_all_data=args['diffusion']['train']['cache_all_data'],
        whole_audio=whole_audio,
        extensions=args['data']['extensions'],
        n_spk=args['common']['n_spk'],
        device=args['diffusion']['train']['cache_device'],
        use_aug=True,
        is_tts = args['diffusion']['model']['is_tts'],
        units_forced_mode = args['data']['units_forced_mode'],
        accelerator=accelerator
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args['diffusion']['train']['batch_size'] if not whole_audio else 1,
        shuffle=True,
        num_workers=args['diffusion']['train']['num_workers'] if args['diffusion']['train']['cache_device'] == 'cpu' else 0,
        persistent_workers=(args['diffusion']['train']['num_workers'] > 0) if args['diffusion']['train']['cache_device'] == 'cpu' else False,
        pin_memory=True if args['diffusion']['train']['cache_device'] == 'cpu' else False
    )
    data_valid = AudioDataset(
        args['data']['valid_path'],
        waveform_sec=args['data']['duration'],
        hop_size=args['data']['block_size'],
        sample_rate=args['data']['sampling_rate'],
        load_all_data=args['diffusion']['train']['cache_all_data'],
        whole_audio=True,
        extensions=args['data']['extensions'],
        n_spk=args['common']['n_spk'],
        is_tts = args['diffusion']['model']['is_tts'],
        units_forced_mode = args['data']['units_forced_mode'],
        accelerator=None
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid


class AudioDataset(Dataset):
    def __init__(
            self,
            path_root,
            waveform_sec,
            hop_size,
            sample_rate,
            load_all_data=True,
            whole_audio=False,
            extensions=['wav'],
            n_spk=1,
            device='cpu',
            use_aug=False,
            is_tts=True,
            units_forced_mode = "nearest",
            accelerator=None
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.units_forced_mode = units_forced_mode
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer = {}
        self.is_tts = is_tts
        self.n_spk = n_spk
        self.spk_name_id_map = {}
        
        if accelerator is not None:
            self.paths = self.paths[accelerator.process_index::accelerator.num_processes]
        
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)

        t_spk_id = 1
        with progress:
            load_task = progress.add_task("Loading", total=len(self.paths))
            for name_ext in self.paths:
                path_audio = os.path.join(self.path_root, 'audio', name_ext)
                duration = librosa.get_duration(path=path_audio, sr=self.sample_rate)
                if not is_tts:
                    path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
                    f0 = np.load(path_f0)
                    f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

                    path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
                    volume = np.load(path_volume)
                    volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

                    path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
                    aug_vol, keyshift = np.load(path_augvol, allow_pickle=True)
                    keyshift = int(keyshift)
                    aug_vol = aug_vol.astype(np.float32)
                    aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                else:
                    f0 = None
                    aug_vol = None
                    volume = None
                    keyshift = 0
    
                if n_spk is not None and n_spk > 1:
                    dirname_split = os.path.dirname(name_ext)
                    if self.spk_name_id_map.get(dirname_split) is None:
                        self.spk_name_id_map[dirname_split] = t_spk_id
                        t_spk_id += 1
                    if t_spk_id < 1 or t_spk_id > n_spk:
                        raise ValueError('[x] spk_id must be a positive integer from 1 to n_spk')
                else:
                    t_spk_id = 1
                spk_id = torch.LongTensor(np.array([t_spk_id])).to(device)

                if load_all_data:
                    path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
                    mel = np.load(path_mel)
                    mel = torch.from_numpy(mel).to(device)
                    if not is_tts:
                        path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
                        aug_mel = np.load(path_augmel)
                        aug_mel = torch.from_numpy(aug_mel).to(device)
                    else:
                        aug_mel = mel
                    path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
                    units = np.load(path_units)
                    units = torch.from_numpy(units).to(device)

                    self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id,
                        't_spk_id': t_spk_id,
                        'keyshift': keyshift
                    }
                else:
                    self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id,
                        't_spk_id': t_spk_id,
                        'keyshift': keyshift
                    }
                progress.update(load_task, advance=1)

    def __getitem__(self, file_idx):
        try:
            name_ext = self.paths[file_idx]
            data_buffer = self.data_buffer[name_ext]
            if data_buffer['duration'] < (self.waveform_sec + 0.1):
                return self.__getitem__((file_idx + 1) % len(self.paths))

            return self.get_data(name_ext, data_buffer)
        except Exception as e:
            return self.__getitem__((file_idx + 1) % len(self.paths))

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug and not self.is_tts

        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel)
            mel = torch.from_numpy(mel).float()

        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units)
            units = units_forced_alignment(units, n_frames=mel.shape[0], units_forced_mode=self.units_forced_mode)
            units = torch.from_numpy(units).float()
        else:
            units = units_forced_alignment(units, n_frames=mel.shape[0], units_forced_mode=self.units_forced_mode)
            units = units[start_frame: start_frame + units_frame_len]
        
        units = units[start_frame: start_frame + units_frame_len, :]
        mel = mel[start_frame: start_frame + units_frame_len, :]

        if not self.is_tts:
            f0 = data_buffer.get('f0')
            aug_shift = 0
            if aug_flag:
                aug_shift = data_buffer.get('keyshift')
            f0_frames = 2 ** (aug_shift / 12) * f0[start_frame: start_frame + units_frame_len]

            vol_key = 'aug_vol' if aug_flag else 'volume'
            volume = data_buffer.get(vol_key)
            volume_frames = volume[start_frame: start_frame + units_frame_len]
            
            aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        else:
            aug_shift = np.array([-1])
            f0_frames = np.array([-1])
            volume_frames = np.array([-1])

        spk_id = data_buffer.get('spk_id')

        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)