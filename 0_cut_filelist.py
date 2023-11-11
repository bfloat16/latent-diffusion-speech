import os
import argparse
from random import shuffle

def traverse_dir(root_dir, extensions,amount=None, str_include=None, str_exclude=None, is_pure=False, is_sort=False, is_ext=True):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        shuffle(file_list)
    return file_list

def split_wav_paths (dir_name, number):
    filelist = traverse_dir(dir_name, extensions=['wav'], is_pure=True, is_sort=True, is_ext=True)

    num_files = len(filelist)
    num_files_per_part = num_files // number
    num_files_leftover = num_files % number

    parts = []
    start = 0
    for i in range(number):
        end = start + num_files_per_part
        if i < num_files_leftover:
            end += 1
        parts.append(filelist[start:end])
        start = end

    for i, part in enumerate(parts):
        if os.path.exists(f"part_{i}.txt"):
            os.remove(f"part_{i}.txt")
        with open(f"part_{i}.txt", "w") as f:
            f.write("\n".join(part))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default="data/train/audio")
    parser.add_argument('--number', type=int, default=20)
    args = parser.parse_args()
    dir_name = args.directory
    number = args.number

    split_wav_paths(dir_name, number)
