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

Nx = 2048
Ny = 2048
aperture_x_1 = 950
aperture_x_2 = 1100
aperture_y_1 = 700
aperture_y_2 = 850



result_matrix = aperture_calc_ints(Nx, Ny, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2)
result_matrix = rescale_matrix(result_matrix, 0, 255)   

plt.figure()
plt.imshow(result_matrix, cmap='viridis', interpolation='nearest')
#plt.show()


np.save(f'/Users/jakubkostial/Documents/phd/code/ff_project/matrix_generation_dmd/repo/newgrating.npy', result_matrix)


