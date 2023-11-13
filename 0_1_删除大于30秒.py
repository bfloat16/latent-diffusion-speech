import os
import wave
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(
    TextColumn("Preprocess:"),
    BarColumn(bar_width=80), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    transient=True
    )

def get_wav_duration(wav_file):
    try:
        with wave.open(wav_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error reading {wav_file}: {e}")
        return None

def delete_short_wav_files(folder_path, tag_duration=30):
    main = rich_progress.add_task("Preprocess", total=len(os.listdir(folder_path)))
    with rich_progress:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_file = os.path.join(root, file)
                    duration = get_wav_duration(wav_file)
                    if duration is not None and duration >= tag_duration:
                        print(f"Deleting {wav_file} (Duration: {duration} seconds)")
                        os.remove(wav_file)
                        lab_file = wav_file.replace(".wav", ".lab")
                        os.remove(lab_file)
            rich_progress.update(main, advance=1)

if __name__ == "__main__":
    folder_path = r'/home/ooppeenn/share/latent-diffusion-speech/data/train/audio'
    delete_short_wav_files(folder_path)