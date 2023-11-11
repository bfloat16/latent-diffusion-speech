import os

def main(data):
    folders = os.listdir(data)
    numbers = sorted([int(folder) for folder in folders])
    i = 1
    for number in numbers:
        if number != i:
            old_path = os.path.join("data/train/audio", str(number))
            new_path = os.path.join("data/train/audio", str(i))
            os.rename(old_path, new_path)
        i += 1

if __name__ == "__main__":
    main("./data/train/audio")