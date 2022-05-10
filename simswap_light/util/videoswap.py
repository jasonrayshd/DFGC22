'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm

from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from parsing_model.model import BiSeNet

from util.norm import SpecificNorm

from util.reverse2original import reverse2wholeimage

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)

    ret = True
    frame_index = 0
    _format = "png"

    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None

    pbar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read() # BGR format
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_results = detect_model.get(frame_rgb, crop_size, stabilize=True, mode="None" if crop_size!=512 else "ffhq")

        if not os.path.exists(temp_results_dir):
            os.mkdir(temp_results_dir)

        if detect_results is not None:
            print(detect_results)
            frame_align_crop = detect_results[0]
            frame_mat = detect_results[1]

            frame_align_crop_tenor = _totensor(frame_align_crop)[None,...].cuda()

            # torch.Tensor, dtype(float32), (ch, h, w), value range in [0, 1]
            swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]

            # cv2.imwrite("output.png",swap_result.cpu().detach().numpy().transpose((1, 2, 0))*255)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.{}'.format(frame_index, _format)), frame)

            reverse2wholeimage(frame_align_crop_tenor, swap_result, frame_mat, crop_size, frame,\
                os.path.join(temp_results_dir, 'frame_{:0>7d}.{}'.format(frame_index, _format)), pasring_model =net,use_mask=use_mask, norm = spNorm)

        else:
            frame = frame.astype(np.uint8)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.{}'.format(frame_index, _format)), frame)
        
        frame_index +=1
        pbar.update(1)

    video.release()

    path = os.path.join(temp_results_dir,f'*.{_format}')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames,fps = fps)
    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    clips.write_videofile(save_path,audio_codec='aac')

