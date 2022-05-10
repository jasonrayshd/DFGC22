import os
import sys
import cv2
import glob
import ffmpeg
from tqdm import tqdm

from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

path = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/result"
dest = "/mnt/traffic/data/competition_datasets/DFGC2022_creationtrack/stage1/compressed"

for folder in tqdm(os.listdir(path)):
    videos = []
    if folder == "id":
        continue
    videos.extend([f"{folder}/{1}/{video}" for video in os.listdir(os.path.join(path, folder, "1"))])
    videos.extend([f"{folder}/{2}/{video}" for video in os.listdir(os.path.join(path, folder, "2"))])
    
    for video in videos:

        video_path = os.path.join(path, video)

        video_forcheck = VideoFileClip(video_path)
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False

        del video_forcheck

        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)

        vcap = cv2.VideoCapture(video_path)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        vcap.release()
        os.system("rm -f ./temp/*")
        os.system(f"ffmpeg -i {video_path} -qscale:v 25  -hide_banner -loglevel error ./temp/frame_%03d.jpg")

        image_filenames = sorted(glob.glob(os.path.join("./temp",'*.jpg')))

        clips = ImageSequenceClip(image_filenames,fps = fps)
        clips = clips.set_audio(video_audio_clip)

        clips.write_videofile(f"{dest}/{video}", audio_codec='aac', codec="libx264")
