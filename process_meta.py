# make id reference image directories for each person
import os
from PIL import Image
import random

def _mkdir(path):
    try:
        os.mkdir(path)
        return 0
    except Exception as e:
        print(e)
        return 1

def mkdir_for_id_ref():

    path = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/metadata_C1.txt"
    dest = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/result/id"

    with open(path, "r") as _fbar:
        for line in _fbar.readlines():
            raw_path = line.strip("\n")
            eC =  _mkdir(os.path.join(dest, raw_path))
            if eC != 0 :
                break

def _pick(path):
    frames = os.listdir(path)
    random.shuffle(frames)
    return frames[0]

def _random_pick():
    path = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/metadata_C1.txt"
    source = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/source/slices/"
    dest = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/result/id"

    with open(path, "r") as _fbar:
        for line in _fbar.readlines():
            raw_path = line.strip("\n")
            im_idx = _pick(os.path.join(source, f"{raw_path}.mp4"))
            out_path = raw_path.split("/")[:-1]
            Image.open(os.path.join(source, f"{raw_path}.mp4", im_idx)).save(os.path.join(dest, *out_path, "id.png"))


if __name__ == "__main__":
    # mkdir_for_id_ref()
    _random_pick()