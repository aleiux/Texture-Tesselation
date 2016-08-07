import numpy as np
import scipy.signal
import scipy.ndimage
import sys
# create a 1 bit mask from the given seam, of size shape
def make_horizontal_mask(seam, shape):
    mask = np.zeros(shape)
    for i in range(0, len(seam)):
        mask[i, seam[i]:] = 1.0
    return mask
    
def calculate_lowest_seam(energy, is_vertical):
    if not is_vertical: #lets go top down (0 and up)
        energy = np.transpose(energy)
    energy_sum = np.zeros(energy.shape)
    right_filter = np.array( [1, 0, 0] ) #what happens if you move right
    left_filter = np.array( [0, 0, 1] ) #what happens if you move left
    center_filter = np.array( [0, 1, 0] ) #what happens if you move center
    energy_sum[0, :] = energy[0, :]
    for row in range(1, energy.shape[0]):
        my_row = energy_sum[row - 1, :]
        right_vals = scipy.ndimage.filters.convolve(my_row, right_filter, mode='constant', cval = sys.float_info.max)
        center_vals = scipy.ndimage.filters.convolve(my_row, center_filter, mode='constant', cval = sys.float_info.max)
        left_vals = scipy.ndimage.filters.convolve(my_row, left_filter, mode='constant', cval = sys.float_info.max)
        lowest_energy = np.minimum(right_vals, center_vals)
        np.minimum(lowest_energy, left_vals, out=lowest_energy)
        #now lowest_energy has the results of the best movement
        energy_sum[row, :] = energy[row, :] + lowest_energy
    path = np.zeros(energy.shape[0])
    path[energy.shape[0] - 1] = np.argmin(energy_sum[energy.shape[0]-1, :])
    for row in range(energy.shape[0] -2, -1, -1):
        prev_col = path[row+1]
        offset = 0
        go_left = sys.float_info.max
        if prev_col > 0:
            go_left = energy_sum[row, prev_col -1]
        go_right = sys.float_info.max
        if prev_col < energy.shape[1] - 1:
            go_right = energy_sum[row, prev_col + 1]
        go_center = energy_sum[row, prev_col]
        if go_left > go_right:
            if go_center > go_right: # right wins
                offset = 1
            else: #center wins
                offset = 0
        else:
            if go_center > go_left: #left wins
                offset = -1
            else: #center wins
                offset = 0
        path[row] = prev_col + offset
    return path