# 2021.12.22 richard
import torch
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from .face_utils import norm_crop, FaceDetector


def deduplicate(landms):
    """
        deduplicate faces detected from Retinaface model
        this is done by claiming that differences between the first point's x value of each landmark of the same face
        should not exceed 50 pixels. Else, it is recognized as two unique faces

        Parameters
        ---
        landms: landmarks of detected face, shape [num_landmarks, 10]. 
        The landmarks indicate 5 points (nose, two eyes, left and right point of mouth) on face
        
        Return
        ---
        tensor of shape (number of unique faces, 10)
    """
    # list of bs numbers of list
    if len(landms) == 0:
        return None
    # sort landmarks
    landms = landms[landms[:, 0].sort()[1]] # [num_landmarks, 10]
    # logging.info(f"landms: {landms}")
    # print(f"landms: {landms} {landms.shape}")
    # compute differences between adjacent x-coordinates
    d = landms[:-1, 0]- landms[1:, 0] # [num_landmarks - 1 ,10]
    # print(f"d: {d} {d.shape}")
    # if the differences between two predictions of landmark are more than 50 pixels
    # then those two predictions should be in different cluster
    p = (d.abs() > 50).nonzero().squeeze() # cutoff points, shape: [num_cutoff_points]
    if p.sum() == 0: # if only one cluster exist
        return landms.mean(dim=0).unsqueeze(0)

    if p.dim() == 0:
        p = torch.tensor([p])

    p = torch.cat((p, torch.tensor([len(landms)-1])), dim = 0)
    # print(f"p: {p} {p.shape}")

    # compute the number of landmarks in each cluster
    d_p = (p[1:] - p[:-1])
    # print(f"d_p: {d_p} {d_p.shape}")
    n = torch.cat((torch.tensor([p[0]+1]), d_p))

    # print(f"n: {n} {n.shape}")
    # average landms for each face detected within a single image
    v = torch.cumsum(landms, dim=0)[p] # shape [num_landmarks, 10]
    s_d = v[1:,:] - v[:-1, :]
    # print(f"v: {v} {v.shape}")
    # print(f"s_d: {s_d} {s_d.shape}")

    summation = torch.cat((v[0].unsqueeze(0), s_d), dim=0) # shape [num_clusters, 10]

    # print(f"summation: {summation} {summation.shape}")
    avg = torch.div(summation, n.unsqueeze(-1).repeat(1, summation.shape[1]))

    return avg

# crop_face, version 1: [For Test] return all faces detected and deduplicate faces for each person
# initialization of facedetector
# facedetector = FaceDetector(device = "cpu", confidence_threshold=0.9)
# facedetector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")

# def crop_face(img, face_detector, imagesize):

#     img = np.array(img)
#     _img = np.copy(img)

#     with torch.no_grad():
#         boxes, landms = face_detector.detect(_img) # time expense 4s

#     landms = deduplicate(landms)
#     if landms == None:
#         return None

#     landms = landms.cpu().numpy().reshape(-1, 5, 2).astype(np.int)
#     imgs = [ norm_crop(np.array(img), landms[i], image_size=imagesize) for i in range(landms.shape[0]) ]
    
#     return imgs

# crop face, version 2: [For Train] return one face with maximum confidence and no deduplicate
# initialization of facedetector
# facedetector = FaceDetector(device = "cpu", _max_inds = True, confidence_threshold=0.9)
# facedetector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")

def crop_face(img, face_detector, image_size, mode="None"):
    """
        Parameters
        ---
        img : pil image or numpy.ndarray, dtype(uint8) in RGB format
            image that detect face from

        Return
        ---
        M : 

        cropped_img : 
    
    """
    img = np.array(img)

    _img = np.copy(img)
    with torch.no_grad():
        _, landms = face_detector.detect(_img) # shape (1,10)

    landmarks = landms.cpu().detach().numpy().reshape(5, 2).astype(np.int) # shape (5, 2)
    M, cropped_img = norm_crop(img, landmarks, image_size=image_size, mode=mode) # shape (h, w, c)

    return M, cropped_img # np.uint8



def get_landms(img, face_detector, _max_inds=False):
    """

        Return
        ---
        landmarks : numpy.ndarray, (number of landmarks, 5, 2), float
    """
    img = np.array(img)
    _img = np.copy(img)

    with torch.no_grad():
        _, landms = face_detector.detect(_img) # shape (1,10)

    landmarks = landms.cpu().detach().numpy().reshape(-1, 5, 2).astype(np.int) # shape (5, 2)

    return landmarks