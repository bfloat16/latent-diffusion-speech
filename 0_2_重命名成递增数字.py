import os

input_dir = "/home/ooppeenn/share/latent-diffusion-speech/data/train/audio"

subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
subdirs.sort()

for i, subdir in enumerate(subdirs):
    new_name = os.path.join(input_dir, str(i+1))
    os.rename(subdir, new_name)