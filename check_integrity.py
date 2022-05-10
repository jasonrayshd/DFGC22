import os

path = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/compressed"

videos = []
for folder in os.listdir(path):
    if folder == "id":
        continue
    videos.extend([f"{folder}/{1}/{video}" for video in os.listdir(os.path.join(path, folder, "1"))])
    videos.extend([f"{folder}/{2}/{video}" for video in os.listdir(os.path.join(path, folder, "2"))])

print(videos)
print(len(videos))