import cv2
import numpy as np
# import  time
import torch
from torch.nn import functional as F
import torch.nn as nn
from util.spline import spline, HRFSspline, GCF

ratio = 0

def encode_segmentation_rgb(segmentation, no_neck=True):
    """
        Return the mask of face area and mouth area in a stacked numpy array.

        Parameters
        ---
        segmentation : torch.Tensor
            predicted segmentation map, shape like (H, W)
        
        no_neck : boolean, optional
            whether there is neck in the image or not. default is True
        
        Return
        ---
            numpy.ndarray, (H, W, 2), value in {0, 255}, which contains map indicates the area of face 
            and mouth respectively.
    
    """
    parse = segmentation

    face_part_ids = [1, 2, 3, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    # hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    eye_map = np.zeros([parse.shape[0], parse.shape[1]])
    nose_map = np.zeros([parse.shape[0], parse.shape[1]])

    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])
    for valid_id in [4, 5, 6]:
        valid_index = np.where(parse==valid_id)
        eye_map[valid_index] = 255
       
    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    nose_map[np.where(parse==10)] = 255
    # valid_index = np.where(parse==hair_id)
    # hair_map[valid_index] = 255
    #return np.stack([face_map, mouth_map,hair_map], axis=2)
    return np.stack([face_map, mouth_map, eye_map, nose_map], axis=2)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask

def draw_mask(a, b):
    """
        draw mask and face
        
        Parameters
        ---
        a : numpy.ndarray, (h, w, 3)
            face image 
        b : numpy.ndarray, (h, w, 3)
            mask image.
    
    """
    import matplotlib.pyplot as plt
    plt.imshow(a)
    plt.imshow(b, alpha=0.5)
    plt.savefig("maskonface.png")

def postprocess(swapped_face, target, target_mask,smooth_mask):
    """

        Parameters:
        ---
        swapped_face : numpy.ndarray, (h, w, ch), dtype(float32) value in [0, 1]

        target : numpy.ndarray, (h, w, ch), dtype(float32, value in [0, 1]

        target_mask : numpy.ndarray, (h, w, 2), value in {0, 255}

        smooth_mask : Class.SoftErosion

        Return:
            result : numpy.ndarray, dtype(float32), (h, w, ch), value in [0, 1]


    """
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))

    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
    face_mask_tensor = mask_tensor[0] + mask_tensor[1] # (h, w)

    """
        DEBUG TEST
        draw target face and mask
    """
    # _face_mask_np = face_mask_tensor.cpu().numpy()[..., np.newaxis] # (h, w, 1)
    # _face_mask_np = np.stack([_face_mask_np, np.zeros_like(_face_mask_np), np.zeros_like(_face_mask_np)], axis=2).squeeze()
    # draw_mask(target,_face_mask_np)

    # step: 1
    # erode the face mask, to make smaller the swapped face area
    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_() # torch.Tensor, (h, w)

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis] # numpy.ndarray, (h, w, 1)

    """
        DEBUG TEST
        draw swapped face and mask
    """
    # _face_mask_np = np.stack([soft_face_mask, np.zeros_like(soft_face_mask), np.zeros_like(soft_face_mask)], axis=2).squeeze()
    # draw_mask(swapped_face, _face_mask_np)

    # step: 2
    # combine swapped face and target face
    result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:,:,::-1]
    return result



def postprocess_wspline(swapped_face, target, target_mask,smooth_mask):
    """
        Parameters:
        ---
        swapped_face : numpy.ndarray, (h, w, ch), RGB format, dtype(float32) value in [0, 1]

        target : numpy.ndarray, (h, w, ch), RGB format, dtype(float32, value in [0, 1]

        target_mask : numpy.ndarray, (h, w, 2), value in {0, 255}

        smooth_mask : Class.SoftErosion

        Return:
            result : numpy.ndarray, dtype(float32), (h, w, ch)
    """
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))
    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
    face_mask_tensor = mask_tensor[0] + mask_tensor[1] # (h, w)

    face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze(0).unsqueeze(0))
    face_mask_np = face_mask_tensor.squeeze(0).repeat_interleave(3, dim=0).cpu().numpy().transpose((1, 2, 0))

    result = HRFSspline(swapped_face*255, target*255, face_mask_np).astype(np.float32) / 255

    return result


def postprocess_wspline_wgcf(swapped_face, target, target_mask,smooth_mask):
    """
        Parameters:
        ---
        swapped_face : numpy.ndarray, (h, w, ch), RGB format, dtype(float32) value in [0, 1]

        target : numpy.ndarray, (h, w, ch), RGB format, dtype(float32, value in [0, 1]

        target_mask : numpy.ndarray, (h, w, 2), value in {0, 255}

        smooth_mask : Class.SoftErosion

        Return:
            result : numpy.ndarray, dtype(float32), (h, w, ch)
    """
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))
    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
    face_mask_tensor = mask_tensor[0] # (h, w)
    mouth_mask_tensor = mask_tensor[1] 
    eye_mask_tensor = mask_tensor[2]
    nose_mask_tensor  = mask_tensor[3]

    face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze(0).unsqueeze(0))
    face_mask_np = face_mask_tensor.squeeze(0).repeat_interleave(3, dim=0).cpu().numpy().transpose((1, 2, 0))
    mouth_mask_np = mouth_mask_tensor.unsqueeze(0).repeat_interleave(3, dim=0).cpu().numpy().transpose((1, 2, 0))
    eye_mask_np = eye_mask_tensor.unsqueeze(0).repeat_interleave(3, dim=0).cpu().numpy().transpose((1, 2, 0))
    nose_mask_np = nose_mask_tensor.unsqueeze(0).repeat_interleave(3, dim=0).cpu().numpy().transpose((1, 2, 0))
    
    global ratio
    if ratio == 0:
        print("ratio is zero")
        ratio = GCF(torch.from_numpy((target*(face_mask_np+mouth_mask_np+eye_mask_np+nose_mask_np)*255).transpose(2, 0, 1)).unsqueeze(0)) / GCF(torch.from_numpy((swapped_face*(face_mask_np+mouth_mask_np+eye_mask_np+nose_mask_np)*255).transpose(2, 0, 1)).unsqueeze(0))
        ratio = ratio.item()

    swapped_face *= ratio
    result = HRFSspline(swapped_face*255, target*255, [face_mask_np, mouth_mask_np, eye_mask_np, nose_mask_np]).astype(np.float32) / 255

    
    return result


def reverse2wholeimage(source_img, swaped_img, mat, crop_size, oriimg, save_path = '', \
                     pasring_model =None,norm = None, use_mask = False):
    """

        Parameters
        ---
        source_imge : torch.Tensor, (bs, ch, h, w), dtype(float32), value range in [0, 1]
            image as input to SimSwap model
        
        swaped_img : numpy.ndarray, (h, w, ch), dtype(float32), value range in [0, 1]
            image as output of SimSwap model

        mat : 
    
    """

    if use_mask:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=20).cuda()
    else:
        pass

    swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0)) # numpy.ndarray, (h, w, ch)
    img_white = np.full((crop_size,crop_size), 255, dtype=float) # numpy.ndarray, (h, w)

    # step: 1
    # inverse the Affine transformation matrix
    mat_rev = np.zeros([2,3])
    div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
    mat_rev[0][0] = mat[1][1]/div1
    mat_rev[0][1] = -mat[0][1]/div1
    mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
    div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
    mat_rev[1][0] = mat[1][0]/div2
    mat_rev[1][1] = -mat[0][0]/div2
    mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

    orisize = (oriimg.shape[1], oriimg.shape[0])

    # step: 2
    # combine input image (target image) and output image (swapped image)
    # target_image is image of area of cropped face with surroundings all qual to zero, of which size is the same with original image

    if use_mask:
        source_img_norm = norm(source_img)                              # torch.Tensor, shape: bs, ch, h, w
        source_img_512  = F.interpolate(source_img_norm,size=(512,512)) # torch.Tensor, shape: bs, ch, h, w
        out = pasring_model(source_img_512)[0]                          # torch.Tensor, shape: bs, 19, h, w
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)       # numpy.ndarray, shape: h, w
        vis_parsing_anno = parsing.copy().astype(np.uint8)
        tgt_mask = encode_segmentation_rgb(vis_parsing_anno)            # numpy.ndarray, shape: h, w, 2, dtype(float64), value in {0, 255}

        if tgt_mask.sum() >= 5000:
            target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))

            # customized
            target_image_parsing = postprocess_wspline_wgcf(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask, smooth_mask)
            # official, raw
            # target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask, smooth_mask)

            target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
            target_image = cv2.cvtColor(target_image.astype(np.float32), cv2.COLOR_RGB2BGR)
        else:
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
    else:
        target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)

    # step: 3
    # prepare mask that is used to combine cropped image and surroundings:
    # a. erosion
    # b. gaussian blurring
    img_white = cv2.warpAffine(img_white, mat_rev, orisize)
    img_white[img_white>20] =255
    img_mask = img_white

    """
        opencv erode(): 
        It needs two inputs, one is original image, second one is called structuring
        element or kernel which decides the nature of operation. A pixel in the original
        image (either 1 or 0) will be considered 1 only if all the pixels under the kernel
        is 1, otherwise it is eroded (made to zero).

        ref: https://www.geeksforgeeks.org/python-opencv-cv2-erode-method/
    """
    kernel = np.ones((40,40),np.uint8)
    img_mask = cv2.erode(img_mask,kernel,iterations = 1) # numpy.ndarray, (h, w) ,binary value in {0, 1}
    
    """
        DEBUG TEST 20220409, Jiachen Lei
        non-zero pixels decrease after the erosion, which indicates the edge of the
        cropped image shrinks

        before erosion:
        print(len(img_white[img_white>0]))  # 495469

        after erosion:
        print(len(img_mask[img_mask>0]))    # 439544
    """

    # step: 4
    # blur the edge of the mask
    kernel_size = (20, 20)
    blur_size = tuple(2*i+1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    img_mask /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1]) # numpy.ndarray, (h, w, 1)

    if use_mask:
        target_image = np.clip(np.array(target_image, dtype=np.float) * 255, 0, 255)
    else:
        target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255

    # step: 5
    # blend cropped image and surroundings
    img = np.array(oriimg, dtype=np.float)
    img = img_mask * target_image + (1-img_mask) * img
    final_img = img.astype(np.uint8)

    cv2.imwrite(save_path, final_img) # save in BGR order
