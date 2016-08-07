import numpy as np
import math
import scipy.signal
import scipy.ndimage

#fore back and mask should be 2D arrays
def blend_channel(fore, back, mask):
    fore_G = generate_gaussian(fore)
    back_G = generate_gaussian(back)
    mask_G = generate_gaussian(mask)
    fore_L = generate_laplacian(fore_G, fore)
    back_L = generate_laplacian(back_G, back)
    #normalize mask_G
    for i in range(0, len(mask_G)):
        mask_G[i] = mask_G[i] / 1.0
    result = np.zeros(mask.shape)
    for i in range(0, len(mask_G)):
        back_contribution = back_L[i] * (1 - mask_G[i])
        fore_contribution = fore_L[i] * mask_G[i]
        result +=  fore_contribution + back_contribution
    lind = len(mask_G) - 1
    result += (fore_G[lind] * mask_G[lind] + back_G[lind] * (1- mask_G[lind]))
    result = np.clip(result, 0, 255)
    return result
    

#generates gaussians from 2D array with sigma = 2, 4, 8, 16... imagesize
def generate_gaussian(image, limit = 16):
    rows = image.shape[0]
    cols = image.shape[1]
    max_sigma = min(rows, cols)
    sigma = 2
    result = []
    while sigma < max_sigma and sigma <= limit:
        result.append(scipy.ndimage.gaussian_filter(image, sigma))
        sigma = sigma * 2
    return result
    
def generate_laplacian(gauss_stack, image):
    stacksize = len(gauss_stack)
    if stacksize == 0:
        print("stacksize is zero")
        return image
    result = [image - gauss_stack[0]]
    i = 0
    while i + 1 < stacksize:
        result.append(gauss_stack[i] - gauss_stack[i+1])
        i+=1
    return result