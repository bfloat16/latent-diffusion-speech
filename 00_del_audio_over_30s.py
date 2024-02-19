import os
import glob
import torchaudio
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def get_wav_duration(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    duration = waveform.size(1) / sample_rate
    return duration

def delete_short_wav_files(folder_path, tag_duration=30):
    wav_files = glob.glob(os.path.join(folder_path, '**/*.wav'), recursive=True)
    main = rich_progress.add_task("Preprocess", total=len(wav_files))
    
    with rich_progress:
        for wav_file in wav_files:
            duration = get_wav_duration(wav_file)
            if duration is not None and duration >= tag_duration:
                lab_file = wav_file.replace(".wav", ".txt")
                os.remove(wav_file)
                os.remove(lab_file)
                print(f"Deleting {wav_file} {lab_file}")
            rich_progress.update(main, advance=1)

if __name__ == "__main__":
    folder_path = r'D:\AI\Project\latent-diffusion-speech\data\train\mihoyo_CN_4.3_1.6_44100'
    delete_short_wav_files(folder_path)