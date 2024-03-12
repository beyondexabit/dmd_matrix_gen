import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm


# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

def aperture_calc_ints(Nx, Ny, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2):
    large_matrix = np.zeros((Nx, Ny))

    aperture = np.ones((aperture_y_2 - aperture_y_1, aperture_x_2 - aperture_x_1))

    large_matrix[aperture_y_1:aperture_y_2, aperture_x_1:aperture_x_2] = aperture
    rotated_arr = np.rot90(large_matrix, k=1)    
    
    return rotated_arr

def aperture_calc(Nx, Ny, extent_x, extent_y, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2):
    large_matrix = np.zeros((Nx, Ny))
    segment_x = aperture_x_2 - aperture_x_1
    segment_y = aperture_y_2 - aperture_y_1
    Nxones = int(Nx * (segment_x / extent_x))
    Nyones = int(Ny * (segment_y / extent_y))

    aperture = np.ones((Nxones, Nyones))

    row_start = int((aperture_x_1 + extent_x/2) / extent_x * Nx)
    col_start = int((aperture_y_1 + extent_y/2) / extent_y * Ny)

    large_matrix[row_start:row_start + Nxones, col_start:col_start + Nyones] = aperture
    rotated_arr = np.rot90(large_matrix, k=1)    
    
    return rotated_arr

extent_x = 30e-3 
extent_y = 30e-3
Nx = 2048
Ny = 2048
'''
aperture_x_1 = -4e-3
aperture_x_2 = 4e-3
aperture_y_1 = -1.5e-3
aperture_y_2 = 1.5e-3
result_matrix = aperture_calc(Nx, Ny, extent_x, extent_y, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2)
result_matrix = rescale_matrix(result_matrix, 0, 255)   
'''

sep = 15
aperture_x_1 = 0
aperture_x_2 = 2048
aperture_y_1 = 1024 - sep
aperture_y_2 = 1024 + sep



result_matrix = aperture_calc_ints(Nx, Ny, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2)
result_matrix = rescale_matrix(result_matrix, 0, 255)   

plt.figure()
plt.imshow(result_matrix, cmap='viridis', interpolation='nearest')
#plt.show()


np.save(f'/Users/jakubkostial/Documents/phd/code/ff_project/matrix_generation_dmd/repo/narrow_slit_30pix.npy', result_matrix)


