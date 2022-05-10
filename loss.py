# Glad to help
# 2022.04.03 Jiachen Lei
# jiachenlei@zju.edu.cn

import math
from matplotlib.pyplot import gcf
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image


def gaussian_filter(kernel_size=11, sigma=1.5, channels = 3, dtype=torch.float64):
    """
        Create gaussian filter

        Parameters
        ---
        kernel_size : int, optional
            kernel size of the gaussian filter, default is 11.

        sigma : float, optional
            variance of gaussian filter, default is 1.5

        channels : int, optional
            channels of guassian filter, default is 3
        
        dtype : torch.dtype, optional
            type of the convolution weights, default is torch.float64

        Return
        ---
        guassian_filter : nn.Module
            guassian filter

        Reference
        ---
        [1] : https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3

    """
    x = torch.arange(kernel_size)
    y = x
    xx, yy = torch.meshgrid(x, y)

    mean = (kernel_size-1)/2
    var = sigma**2

    # p(x,y) = p(x)*p(y)
    _kernel = ( 1/(2*math.pi*sigma) ) * (torch.exp( - ( (xx-mean)**2 + (yy-mean)**2) / (2*var) ) )
    _kernel = _kernel / _kernel.sum()
    _kernel = _kernel.view(1, 1, kernel_size, kernel_size)
    _kernel = _kernel.repeat(channels, 1, 1, 1)

    _kernel = _kernel.to(dtype)

    guassian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, groups=channels, bias=False)
    guassian_filter.weight.data = _kernel
    guassian_filter.weight.requires_grad = False

    return guassian_filter


def MSSIM(output, target, L=255, const=[0.01, 0.03], weights=[1,1,1]):

    """
        Mean structural similarity measure, for more information refer to [paper](1)
        
        The result of this implementation is slightly different from the official matlab
        implementation due to the reason of precision. Matlab default precision is 16-bit
        while pytorch usually uses 32-bit or 64-bit precision.

        Parameters
        ---
        output : torch.Tensor
            output of model

        target: torch.Tensor
            ground truth

        L: int, optional
            dynamic range of pixel values, default is 255. should be set to 1 when the output/target is 
            normalized to [0, 1], or set to 2 when normalized to [-1, 1]

        const: list, optional
            values of constants K1 and K2, default is 0.01 and 0.03

        weights: list, optional
            weights of luminance, contrast, structural loss.

        Return
        ---
        loss : torch.Tensor
            loss measured in SSIM

    """

    assert output.shape == target.shape, "shape of output mismatches the one of target."
    assert output.dtype == target.dtype, "dtype of output mismatches the one of target."

    K1, K2 = const
    C1, C2 = (K1*L)**2, (K2*L)**2
    C3 = 0.5*C2
    channels = output.shape[1]
    dtype = output.dtype

    gausf = gaussian_filter(kernel_size=11, sigma=1.5, channels=channels, dtype=dtype)

    miu1 = gausf(output)
    miu2 = gausf(target)
    sigma1 = torch.sqrt( torch.clamp(gausf(output * output) - miu1*miu1, min=0) )
    sigma2 = torch.sqrt( torch.clamp(gausf(target * target) - miu2*miu2, min=0) )
    sigma12 = torch.clamp(gausf(output * target) - miu1*miu2, min=0) 

    l = (2*miu1*miu2 + C1)/(miu1**2 + miu2**2 + C1)
    c = (2*sigma1*sigma2 + C2)/(sigma1**2 + sigma2**2 + C2)
    s = (sigma12 + C3)/(sigma1*sigma2 + C3)

    alpha, beta, lamda = weights
    ssim = l.pow(alpha)*c.pow(beta)*s.pow(lamda)

    # compute mean ssim
    # divide before summation avoiding overflow
    mssim= ssim / (ssim.shape[1] * ssim.shape[2] * ssim.shape[3])
    mssim = mssim.to(dtype)
    mssim = mssim.sum()

    return mssim


def debug_ssim():
    import numpy as np
    target = torch.from_numpy(np.array(Image.open("n000002_0001_01-n000372_0161_01.jpg"))).transpose(0, 2).transpose(1, 2) / 1
    output = torch.from_numpy(np.array(Image.open("n000002_0001_01-n000372_0161_01.jpg"))).transpose(0, 2).transpose(1, 2) / 1
    
    target = target.to(torch.float64)
    output = output.to(torch.float64)
    
    output += 0.01*255
    # output = (output - 0.5) /0.5
    # target = (target - 0.5) /0.5

    # print(output)
    ssim = MSSIM(output[1, :, :].view(1, 1, *output.shape[1:]), target[1, :, :].view(1, 1, *output.shape[1:]), L=255)
    print(ssim)


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
        output : torch.Tensor
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


def debug_GCF():
    from PIL import Image
    import numpy as np
    input = torch.from_numpy(np.array(Image.open("n000002_0001_01-n000372_0161_01.jpg"))).transpose(0,2).transpose(1,2)[:, :224, :224]
    input = input.to(torch.double)
    gcf = GCF(input.unsqueeze(0))

    print(gcf)

if __name__ =="__main__":
    debug_GCF()