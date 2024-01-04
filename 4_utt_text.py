import os
from glob import glob
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

def main(input_dir):
    main = rich_progress.add_task("Preprocess", total=len(os.listdir(input_dir)))
    with rich_progress:
        for subdir in os.listdir(input_dir):
            subdir_path = os.path.join(input_dir, subdir)
            utt_txt_file = os.path.join(subdir_path, 'utt_text.txt')
            if os.path.exists(utt_txt_file):
                os.remove(utt_txt_file)

            lab_files = glob(os.path.join(subdir_path, '*.txt'))
            for lab_file in lab_files:
                wav_file = lab_file.replace('.txt', '.wav')

                with open(lab_file, 'r', encoding='utf-8') as lab:
                    lab_text = lab.read()
                
                with open(os.path.join(subdir_path, 'utt_text.txt'), 'a', encoding='utf-8') as utt_txt:
                    utt_txt.write(f'{os.path.basename(wav_file)}|{lab_text}\n')
            rich_progress.update(main, advance=1)

if __name__ == '__main__':
    main(r"D:\AI\Project\latent-diffusion-speech\data\train\audio")
    main(r"D:\AI\Project\latent-diffusion-speech\data\val\audio")