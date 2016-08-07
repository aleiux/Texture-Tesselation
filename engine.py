import numpy as np
import scipy.ndimage
import scipy.interpolate
import math
import sys
import os
import re
import cv2
import seam_util
import blend_util
import random

#parse and convert into row, column (given that the input is x, y). conv_shape is r/c
def parse_swap(filename, conv_shape):
    if not os.path.isfile(filename):
        return []
    p = re.compile("^[0-9 \.]{2,}?$", re.M)
    results = p.findall(open(filename).read())
    for i in range(0, len(results)):
        tokens = results[i].split(' ')
        c_rel = float(tokens[0])
        r_rel = float(tokens[1])
        results[i] = (r_rel * conv_shape[0], c_rel * conv_shape[1])
    return results

def validate(image):
    image = np.clip(image, 0, 255)
    return image.astype("uint8")

#takes largest r/c and smallest r/c values to be the boundaries for the rectangle
#aspect ratio = width / height (or cols / rows)
def extract_region(image, vertices, aspect_ratio = 1.0):
    vertices = np.asarray(vertices)
    row_coordinates = vertices[:,0]
    col_coordinates = vertices[:, 1]
    row_min = row_coordinates.min()
    col_min = col_coordinates.min()
    row_max = row_coordinates.max()
    col_max = col_coordinates.max()
    assert row_min >= 0 and col_min >= 0
    assert row_max < image.shape[0] and col_max < image.shape[1]
    dr = row_max - row_min
    dc = col_max - col_min
    if aspect_ratio * dr > dc: #too many rows
        dr = int(dc / aspect_ratio)
    else: #too many columns
        dc = int(aspect_ratio * dr)
    return image[row_min:row_min + dr, col_min:col_min + dc, :]

# generates spline function for 2d gray image array
def generate_spline_function(image, super_rows, super_cols):
    z = np.zeros((super_rows, super_cols))
    rows = image.shape[0]
    cols = image.shape[1]
    patch_rows = int( math.ceil(rows / super_rows) )
    patch_cols = int( math.ceil(cols / super_cols) )
    rowmid = int(patch_rows / 2)
    colmid = int(patch_cols / 2)
    r = range(rowmid, (super_rows) * patch_rows, patch_rows)
    c = range(colmid, (super_cols) * patch_cols, patch_cols)
    assert len(r) == super_rows and len(c) == super_cols
    for i in range(0, len(r)):
        for j in range(0, len(c)):
            row_start = (i - 0.1) *patch_rows
            row_end = (i+1.1) * patch_rows
            col_start = (j - 0.1) *patch_cols
            col_end = (j+1.1) * patch_cols
            row_start = int(max(row_start, 0))
            col_start = int(max(col_start, 0))
            row_end = int(min(row_end, rows))
            col_end = int(min(col_end, cols))
            average = np.average(image[row_start:row_end, col_start:col_end])
            z[i, j] = average
    func = scipy.interpolate.RectBivariateSpline(r, c, z, bbox = [0, rows, 0, cols], kx = 1, ky = 1)
    return func
def evaluate_spline_function(spline, shape):
    rows = range(0, shape[0])
    cols = range(0, shape[1])
    return spline(rows, cols)
    
#normalize lighting effects
def light_normalize_pass(image):
    blue = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)
    blue_filter = None
    green_filter = None
    red_filter = None
    rows = image.shape[0]
    cols = image.shape[1]
    size = 150.0 #roughly one spline point per 150x150 pixel patch
    min_dim = 400
    super_rows = int( round(rows / size) )
    super_cols = int( round(cols / size) )
    if super_rows < 4 or super_cols < 4:
        super_rows = 4
        super_cols = 4
    if rows < min_dim or cols < min_dim:
        print("not large enough for spline lighting adjustment")
        sigma = 50
        blue_filter = scipy.ndimage.filters.gaussian_filter(blue, sigma, mode = 'reflect')
        green_filter = scipy.ndimage.filters.gaussian_filter(green, sigma, mode = 'reflect')
        red_filter = scipy.ndimage.filters.gaussian_filter(red, sigma, mode = 'reflect')
        np.clip(blue_filter, 1, 255, out=blue_filter)
        np.clip(green_filter, 1, 255, out=green_filter)
        np.clip(red_filter, 1, 255, out=green_filter)
        blue_mean = np.average(blue_filter)
        green_mean = np.average(green_filter)
        red_mean = np.average(red_filter)
    else:
        spline_b = generate_spline_function(blue, super_rows, super_cols)
        spline_g = generate_spline_function(green, super_rows, super_cols)
        spline_r = generate_spline_function(red, super_rows, super_cols)
        blue_filter = evaluate_spline_function(spline_b, image.shape)
        green_filter = evaluate_spline_function(spline_g, image.shape)
        red_filter = evaluate_spline_function(spline_r, image.shape)
        np.clip(blue_filter, 1, 255, out=blue_filter)
        np.clip(green_filter, 1, 255, out=green_filter)
        np.clip(red_filter, 1, 255, out=red_filter)
        blue_mean = np.average(blue_filter)
        green_mean = np.average(green_filter)
        red_mean = np.average(red_filter)
        blue_std = np.std(blue_filter)
        green_std = np.std(green_filter)
        red_std = np.std(red_filter)
        np.clip(blue_filter, blue_mean - 2 * blue_std, blue_mean + 2 * blue_std, out=blue_filter)
        np.clip(green_filter, green_mean - 2 * green_std, green_mean + 2 * green_std, out=green_filter)
        np.clip(red_filter, red_mean - 2 * red_std, red_mean + 2 * red_std, out=red_filter)
    #blue, green, red filters at this point represent the illumination at that point
    blue_filter = blue_mean / blue_filter #not the right name, but we'll keep mem
    green_filter = green_mean / green_filter
    red_filter = red_mean / red_filter
    blue = blue * blue_filter
    red = red * red_filter
    green = green * green_filter
    return np.dstack((blue, green, red))

def light_normalize(image):    
    return light_normalize_pass(image)

#tesselate via reflection 
def texture_reflect(image):
    blue = image[:, :, 0] #let these be top left
    green = image[:, :, 1]
    red = image[:, :, 2]
    tr_blue = np.fliplr(blue) #top right
    tr_green = np.fliplr(green)
    tr_red = np.fliplr(red)
    tr_blue = np.delete(tr_blue, 0, axis = 1)
    tr_green = np.delete(tr_green, 0, axis = 1)
    tr_red = np.delete(tr_red, 0, axis = 1)
    final_blue = np.concatenate((blue, tr_blue), axis = 1)
    final_green = np.concatenate((green, tr_green), axis = 1)
    final_red = np.concatenate((red, tr_red), axis = 1)
    final_blue = np.concatenate((final_blue, np.delete(np.flipud(final_blue), 0, axis = 0)), axis = 0)
    final_red = np.concatenate((final_red, np.delete(np.flipud(final_red), 0, axis = 0)), axis = 0)
    final_green = np.concatenate((final_green, np.delete(np.flipud(final_green), 0, axis = 0)), axis = 0)
    return np.dstack((final_blue, final_green, final_red))

def horizontal_quilt(image, OV_FACTOR = 0.15):
    rows = image.shape[0]
    cols = image.shape[1]
    overlap_amount = int(cols * OV_FACTOR)
    safe_stack = image[:, overlap_amount : cols - overlap_amount, :]
    left_stack = image[:, 0 : overlap_amount, :]
    right_stack = image[:, cols - overlap_amount : cols, :]
    assert left_stack.shape == right_stack.shape
    #energy function is absolute difference between left and right 
    energy = np.absolute(np.sum(left_stack - right_stack, axis = 2))
    safe_zone = 16
    if safe_zone * 2 > overlap_amount:
        print("image too small for safe zone. ignoring")
    else:
        energy[:, :safe_zone] *= 2.0
        energy[:, overlap_amount - safe_zone:] *= 2.0
    
    #add corner weights to make top/down and left/right consistent
    corner_weight = -1 * np.average(energy) * rows * 0.5
    energy[0, overlap_amount-1] = corner_weight
    energy[rows-1, overlap_amount-1] = corner_weight
    
    seam = seam_util.calculate_lowest_seam(energy, True) #lowest vertical seam
    mask = seam_util.make_horizontal_mask(seam, energy.shape)
    left_b = left_stack[:, :, 0]
    left_g = left_stack[:, :, 1]
    left_r = left_stack[:, :, 2]
    right_b = right_stack[:, :, 0]
    right_g = right_stack[:, :, 1]
    right_r = right_stack[:, :, 2]
    left_b = blend_util.blend_channel(left_b, right_b, mask) #reuse left
    right_b = None
    left_g = blend_util.blend_channel(left_g, right_g, mask)
    right_g = None
    left_r = blend_util.blend_channel(left_r, right_r, mask)
    right_r = None
    left_stack = np.dstack((left_b, left_g, left_r))
    image = np.concatenate((left_stack, safe_stack), axis = 1)
    return image
    
#tesselate via lowest energy seam
def texture_quilt(image):
    image = horizontal_quilt(image)
    image = image.transpose((1, 0, 2))
    image = horizontal_quilt(image)
    image = image.transpose((1, 0, 2))
    return image

#simple multiresolution blending
def horizontal_blend(image, OV_FACTOR = 0.1):
    cols = image.shape[1]
    overlap_amount = int(cols * OV_FACTOR)
    safe_stack = image[:, overlap_amount : cols - overlap_amount, :]
    left_stack = image[:, 0 : overlap_amount, :]
    right_stack = image[:, cols - overlap_amount : cols, :]
    assert left_stack.shape == right_stack.shape 
    mask = np.zeros((left_stack.shape[0], left_stack.shape[1]))
    mask[:, overlap_amount/2:] = 1.0 # mask is same for r, g, b
    left_b = left_stack[:, :, 0]
    left_g = left_stack[:, :, 1]
    left_r = left_stack[:, :, 2]
    right_b = right_stack[:, :, 0]
    right_g = right_stack[:, :, 1]
    right_r = right_stack[:, :, 2]
    left_b = blend_util.blend_channel(left_b, right_b, mask) #reuse left
    right_b = None
    left_g = blend_util.blend_channel(left_g, right_g, mask)
    right_g = None
    left_r = blend_util.blend_channel(left_r, right_r, mask)
    right_r = None
    left_stack = np.dstack((left_b, left_g, left_r))
    image = np.concatenate((left_stack, safe_stack), axis = 1)
    return image
#perform horizontal blend twice, once vertically and once normally
def texture_blend(image):
    image = horizontal_blend(image)
    image = image.transpose((1, 0, 2))
    image = horizontal_blend(image)
    image = image.transpose((1, 0, 2))
    return image
    #reminder to self. white is foreground

def normalize_matrix(H):
    return H/H[2, 2]    
def transnorm(H, vector):
    vector = np.dot(H, vector)
    scale = vector[2]
    if scale < 0.0001 and scale > -0.0001:
        if scale == 0:
            return vector
    return vector/scale
def swap_matrix_order(H):
    a = H[0,0]
    b = H[0,1]
    c = H[0,2]
    d = H[1,0]
    e = H[1,1]
    f = H[1,2]
    g = H[2,0]
    h = H[2,1]
    return np.array([[e, d, f], [b, a, c], [h, g, 1]])
    
# returns 3x3 tesselation
def example_tesselation(image):
    image = scipy.misc.imresize(image.astype("uint8"), 0.333)
    side = np.concatenate((image, image, image) , 0) #3 by 1
    return np.concatenate((side, side, side), 1) #3 by 3
