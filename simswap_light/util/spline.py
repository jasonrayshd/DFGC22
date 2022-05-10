import cv2
import torch
from torch import nn

import numpy as np
from PIL import Image
from skimage.util import view_as_blocks


def gaussian_pyramid(path, level):
    """
        Return the gaussian pyramid with specified levels of a given image
        The size of each level in the pyramid shrinks with index

        Parameters
        ---
        path : str or numpy.ndarray
            Path of the image or image data in numpy.ndarray

        level : int
            Total level of the pyramid 

        Return
        ---
        pyramid : list
            list contains the pyramid

    """
    if isinstance(path, str):
        bgr_img = cv2.imread(path) # B G R
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # R G B
    elif isinstance(path, np.ndarray):
        img = path
    else:
        raise TypeError('Expect str or numpy.ndarray, but {} was given'.format(type(path)))

    img = img.astype(np.float64) # this will improve precision of reconstructed image

    pyramid = [img]
    for i in range(level):
        img = cv2.pyrDown(img)
        pyramid.append(img)

    return pyramid


def laplacian_pyramid(path, level):
    """
        Return the laplacian pyramid with specified levels of a given image
        the size of each level in the pyramid grows with index.

        Observation
        ---
        Theoretically, laplacian pyramid can be used to reconstruct the original image losslessly, however,
        due to the limited bytes that stores image pixels, it is lossy in practice. We can use np.float64 to
        achieve lossless reconstruction with laplacian pyramid.

        Parameters
        ---
        path : str or numpy.ndarray
            Path of the image or image data in numpy.ndarray

        level : int
            Total level of the pyramid 

        Return
        ---
        pyramid : list
            list contains the pyramid
    """

    gaus_pyr = gaussian_pyramid(path, level)
    pyramid = [gaus_pyr[level-1]]

    for i in range(level-1, 0, -1):
        GE = cv2.pyrUp(gaus_pyr[i])
        L = cv2.subtract(gaus_pyr[i-1], GE)
        pyramid.append(L)

    return pyramid


def lapyr2img(la_pyr):
    """
        Convert laplacian pyramid to original image

        Parameters
        ---
        la_pyr : list
            laplacian pyramid

        Return
        ---
        img : numpy.ndarray
            recovered image
    """
    level = len(la_pyr)
    img = la_pyr[0]
    for i in range(1, level):
        img = cv2.pyrUp(img)
        img = cv2.add(img, la_pyr[i])

    return img


def spline(spath, dpath, mask, level=6):
    """
        Perform image splining and combine certain area on source image specified by 
        mask with target image

        Parameters
        ---
        spath : str or numpy.ndarray
            Path of the 1st image or image data in numpy.ndarray

        dpath : str or numpy.ndarray
            Path of the 2nd image or image data in numpy.ndarray

        mask : str or numpy.ndarray
            Path of mask image or mask image data in numpy.ndarray

        level : int
            Number of levels of pyramid

        Return
        ---
        img : numpy.ndarray
            Splined image
    
    """
    source_lapyr = laplacian_pyramid(spath, level)
    dest_lapyr = laplacian_pyramid(dpath, level)
    mask_lapyr = gaussian_pyramid(mask, level)
    
    lapyr = []
    for i in range(level):
        lapyr.append(mask_lapyr[level-i-1]*source_lapyr[i]+(1-mask_lapyr[level-i-1])*dest_lapyr[i])

    img = lapyr2img(lapyr)

    return img


def HRFSspline(spath, dpath, mask, level=6):
    """
        Perform image splining and combine certain area on source image specified by 
        mask with target image

        Parameters
        ---
        spath : str or numpy.ndarray
            Path of the 1st image or image data in numpy.ndarray

        dpath : str or numpy.ndarray
            Path of the 2nd image or image data in numpy.ndarray

        mask : str or numpy.ndarray
            Path of mask image or mask image data in numpy.ndarray

        level : int
            Number of levels of pyramid

        Return
        ---
        img : numpy.ndarray
            Splined image
    
    """
    source_lapyr = laplacian_pyramid(spath, level)
    dest_lapyr = laplacian_pyramid(dpath, level)
    # mask_lapyr = gaussian_pyramid(mask, level)

    face_mask_np ,mouth_mask_np, eye_mask_np, nose_mask_np = mask
    face_mask_lapyr = gaussian_pyramid(face_mask_np, level)
    mouth_mask_lapyr = gaussian_pyramid(mouth_mask_np, level)
    eye_mask_lapyr = gaussian_pyramid(eye_mask_np, level)
    nose_mask_lapyr = gaussian_pyramid(nose_mask_np, level)
    
    lapyr = []
    for i in range(level):
        lapyr.append((1-face_mask_lapyr[level-i-1] - mouth_mask_lapyr[level-i-1] - eye_mask_lapyr[level-i-1] - nose_mask_lapyr[level-i-1])*dest_lapyr[i])
        if i <= 2:
            lapyr[i] += ((face_mask_lapyr[level-i-1])*dest_lapyr[i]) + (nose_mask_lapyr[level-i-1] + mouth_mask_lapyr[level-i-1] + eye_mask_lapyr[level-i-1])*source_lapyr[i]
        else:
            lapyr[i] += ((face_mask_lapyr[level-i-1] + mouth_mask_lapyr[level-i-1] + eye_mask_lapyr[level-i-1] + nose_mask_lapyr[level-i-1])*source_lapyr[i])

    img = lapyr2img(lapyr)

    return img


def _extrapolate(arr, x, y):
    """
        Extrapolate array by row or column with reflection and inversion.
        
        e.g. 
            gfedcb|abcdefgh|gfedcba
        
        where | represents edge and characters between | are original content

        Parameters
        ---
        arr : numpy.ndarray
            array to be extrapolated

        x : int
            raw index to be extrapolated. if is -1, then extrapolate every element according to y

        y : int
            column index to be extrapolated. if is -1, then extrapolate every element according to x

    """
    h, w, ch = arr.shape

    if x == -1 and y == -1:
        raise ValueError('Expect input are x=-1, y>0 or x>0 x=-1')

    if x == -1:
        int_arr = np.zeros((h, w+1, ch))
        int_arr[:, :-1, :] = arr
        int_arr[:, y, :] = arr[:, 2*(w-1)-y, : ]
    elif y == -1:
        int_arr = np.zeros((h+1, w, ch))
        int_arr[:-1, :, :] = arr
        int_arr[x, :, :] = arr[2*(h-1)-x, :, : ]

    return int_arr


def non_overlap_spline(path, block=8, level=2):
    """
        Perform non-overlap image splining and spline adjacent blocks of pixels in given image

        2022.04.02, Jiachen Lei
        Notice that in the [paper](1), it enlarge 16x16 block into 17x17 by extraplolating at right and 
        bottom side of the block. However, in the downsampling operation of pyramid, 17x17 block will be
        downsampled to 9x9 , but in the upsampling operation of pyramid, 9x9 block will be upsampled to 
        18x18. Thus, instead of enlarging block by 1 pixel at row and column, I choose to enlarge it by 2.

        Parameters
        ---
        path : str or numpy.ndarray
            path of image or image data in numpy.ndarray

        block : int
            block size, the block is assumed to be a square

        level : int
            number of levels of pyramid

        Return
        ---
        img : numpy.ndarray
            splined image

        Reference
        ---
        [1] : A Multiresolution Spline With Application to Image Mosaics 
    
    """
    if isinstance(path, str):
        arr_im = np.array(Image.open(path))
    elif isinstance(path, np.ndarray):
        arr_im = path

    h, w, ch = arr_im.shape
    block_num = h // block

    blocks = view_as_blocks(arr_im, block_shape=(16, 16, 3)).squeeze() # 64, 64, 16, 16, 3 if input is 1024,1024,3

    lst_lapyr = []
    for i in range(block_num):
        for j in range(block_num):
            arr_block = blocks[i, j]

            if i != block_num -1 and j != block_num - 1:
                # only extrapolate blocks that are not at the right or bottom side
                # extrapolate the block at right, bottom side
                # extrapolate two rows/columns in total
                arr_block = _extrapolate(arr_block, block, -1)
                arr_block = _extrapolate(arr_block, block+1, -1)
                arr_block = _extrapolate(arr_block, -1, block)
                arr_block = _extrapolate(arr_block, -1, block+1)

            lapyr = laplacian_pyramid(arr_block, level = level)
            lst_lapyr.append(lapyr)

        rst_img_lapyr.append(lapyr)
    
    splined_im = np.zeros((h, w, ch))

    return rst_img


def debug_recons():
    path = "/mnt/traffic/data/deepfake/Synthesis/FFHQ/faceshifter/images1024x1024/00000/00000_00168-59000_59735.png"
    level = 5
    la_pyr = laplacian_pyramid(path, level)
    import matplotlib.pyplot as plt

    # print(la_pyr)
    for i in range(level):
        plt.subplot(1, level, i+1)
        plt.imshow(la_pyr[i].astype(np.uint8))

    plt.savefig("la_pyr.png")

    img = lapyr2img(la_pyr)
    plt.figure()
    plt.imshow(img)
    plt.savefig("recovered_image.png")

    plt.figure()
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.savefig("image.png")

    print(np.sum(img - cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)))


def debug_spline():
    spath = "/mnt/traffic/data/deepfake/Synthesis/FFHQ/faceshifter/images1024x1024/00000/00000_00168-59000_59735.png"
    tpath = "/mnt/traffic/data/deepfake/Synthesis/FFHQ/faceshifter/images1024x1024/00000/00000_00015-31000_31818.png"
    level = 6

    mask = np.zeros((1024,1024,3))
    mask[250:750, 250:750, :] = 1

    img = spline(spath, tpath, mask, level)

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.savefig("spline.png")


def debug_non_overlap_spline():
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt
    spath = "/mnt/traffic/data/deepfake/Synthesis/FFHQ/faceshifter/images1024x1024/00000/00000_00168-59000_59735.png"
    # im = Image.open(spath)
    # arr_im = np.array(im)
    # im.save("Jpeg_comp.jpg", quality=10, subsampling=0)

    # im_comp = Image.open("Jpeg_comp.jpg")
    # arr_im_comp = np.array(im_comp)

    # # before splinnig
    # print(np.sum(abs(arr_im-arr_im_comp)))

    non_overlap_spline(spath, 16, 2)


def super_pixel(linear_lum):
    """
        generate super pixel with 4 neighboring pixels

        Parameters
        ---
        linear_lum : torch.Tensor
            linear luminance

    """
    _kernel = (1/4) * torch.ones((1, 1, 2, 2))
    _kernel = _kernel.to(torch.double)
    filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, groups=1, bias=False)
    filter.weight.data = _kernel
    filter.weight.requires_grad = False

    super_lum = filter(linear_lum)

    return super_lum


def GCF(output, level=9):
    """
        Global contrast factor, refer to [paper](1) for more details.
        By default, the function will end if the image size is 1x1.

        Parameters
        ---
        output : torch.Tensor, value range in [0, 255]
            output of model

        level : int, optional
            resolution level. default is 9

        Reference
        ---
        [1] : Global Contrast Factor - a New Approach to Image Contrast
    """

    assert output.shape[1] in [1, 3], "Can only compute GCF for 1 or 3 channel images."

    bs, _, h, w = output.shape
    if output.shape[1] == 3:
        gray_output = output[:, 0,]*0.299 + output[:, 1,]*0.587 + output[:, 2,]*0.114
    else:
        gray_output = output

    gray_output = gray_output.view(bs, 1, h, w)
    linear_lum = (gray_output/255).pow(2.2).view(bs, 1, h, w).double()
    gcfs=[0 for i in range(level)]

    for k in range(level):
        percep_lum = 100 * linear_lum.sqrt()
        h, w= percep_lum.shape[2], percep_lum.shape[3]

        # need further optimization
        for i in range(h):
            for j in range(w):
                cfs = abs(percep_lum[:, 0, i, j] - percep_lum[:, 0, i+1,j] if i+1 < h else percep_lum[:, 0, i, j]) + \
                abs(percep_lum[:, 0, i, j] - percep_lum[:, 0, i-1,j] if i-1 < 0 else percep_lum[:, 0, i, j]) + \
                abs(percep_lum[:, 0, i, j] - percep_lum[:, 0, i,j+1] if j+1 > w else percep_lum[:, 0, i, j]) + \
                abs(percep_lum[:, 0, i, j] - percep_lum[:, 0, i,j-1] if j-1 < 0 else percep_lum[:, 0, i, j])
                cfs /= 4
            gcfs[k] += cfs

        gcfs[k] /= h*w
        w_i = (-0.406385*(k+1)/9 + 0.334573)*(k+1)/9 + 0.0877526
        gcfs[k] *= w_i

        if k == level-1 or h == 1 or w == 1:
            break

        linear_lum = super_pixel(linear_lum)

    return sum(gcfs)

if __name__ == "__main__":
    debug_non_overlap_spline()