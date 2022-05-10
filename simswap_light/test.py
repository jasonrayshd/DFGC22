'''
Glad to help
2022.04.09, Jiachen Lei
jiachenlei@zju.edu.cn
This file referred to "https://github.com/neuralchen/SimSwap"
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os



def lcm(a, b):
    """
        Find least common multiple of given integers

        Parameters
        ---
        a : int
            1st integer
        b : int
            2nd integer

        Return
        ---
        int, least common multiple of given integers
    """
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0

# transformation for face detection
transformer_Arcface = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    opt = TestOptions().parse()
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    # option used:
    # 
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    with torch.no_grad():
        # step 1:
        # detect face from given image that provides identity information
        pic_a = opt.pic_a_path
        img_a_whole = cv2.cvtColor(cv2.imread(pic_a), cv2.COLOR_BGR2RGB)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size, stabilize=True, mode="None" if crop_size!=512 else "ffhq")

        # cv2.imwrite("target_face.png", cv2.cvtColor(img_a_align_crop, cv2.COLOR_RGB2BGR))

        img_a = transformer_Arcface(img_a_align_crop)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        img_id = img_id.cuda()

        # step 2:
        # create identity latent feature from the cropped face image
        img_id_downsample = F.interpolate(img_id, size=(112,112)) # down/upsample to 112x112
        latend_id = model.netArc(img_id_downsample)
        latend_id =  F.normalize(latend_id, p=2, dim=1)
 
        # step 3:
        # start face swapping
        video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)