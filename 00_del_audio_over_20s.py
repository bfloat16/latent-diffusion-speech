import os
import glob
import torchaudio
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(
    TextColumn("Preprocess:"),
    BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn()
    )

def get_wav_duration(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    duration = waveform.size(1) / sample_rate
    return duration

def delete_short_wav_files(folder_path, tag_duration=20):
    wav_files = glob.glob(os.path.join(folder_path, '**/*.wav'), recursive=True)
    main = rich_progress.add_task("Preprocess", total=len(wav_files))
    
    with rich_progress:
        for wav_file in wav_files:
            duration = get_wav_duration(wav_file)
            if duration is not None and duration >= tag_duration:
                print(f"Deleting {wav_file}")
                os.remove(wav_file)
                lab_file = wav_file.replace(".wav", ".lab")
                os.remove(lab_file)
            rich_progress.update(main, advance=1)

if __name__ == "__main__":
    folder_path = r'data\train\audio'
    delete_short_wav_files(folder_path)