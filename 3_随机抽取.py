import os
import random
import shutil

train_dir = r"D:\AI\Project\latent-diffusion-speech\data\train\audio"
val_dir = r"D:\AI\Project\latent-diffusion-speech\data\val\audio"

subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

dir_file_counts = [(d, len(os.listdir(d))) for d in subdirs]
sorted_subdirs = sorted(dir_file_counts, key=lambda x: x[1], reverse=True)
sorted_subdirs = [d for d, _ in sorted_subdirs[:5]]

for subdir in sorted_subdirs:
    audio_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".wav") and os.path.isfile(os.path.join(subdir, f))]
    random_audio_files = random.sample(audio_files, 3)
    
    for audio_file in random_audio_files:
        dest_dir = os.path.join(val_dir, os.path.relpath(subdir, train_dir))
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(audio_file, dest_dir)

        lab_file = audio_file.replace(".wav", ".txt")
        if os.path.exists(lab_file):
            shutil.move(lab_file, dest_dir)