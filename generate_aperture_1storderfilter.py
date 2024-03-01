import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm


# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

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
aperture_x_1 = 1e-3
aperture_x_2 = 3.5e-3
aperture_y_1 = -1.5e-3
aperture_y_2 = 1.5e-3

result_matrix = aperture_calc(Nx, Ny, extent_x, extent_y, aperture_x_1, aperture_x_2, aperture_y_1, aperture_y_2)
result_matrix = rescale_matrix(result_matrix, 0, 255)   
np.save(f'/Users/jakubkostial/Documents/phd/code/ff_project/matrix_generation_dmd/repo/dmd_matrix_gen/masks/testgrating_ti.npy', result_matrix)




sys.exit()






# Place the rescaled matrix in the center of the larger matrix






large_matrix[center_y:center_y + matrix_new.shape[0], center_x:center_x + matrix_new.shape[1]] = matrix_new

large_matrix = rescale_matrix(large_matrix, 0, 255)


plt.figure()
plt.imshow(large_matrix, cmap='viridis', interpolation='nearest')
plt.show()


np.save(f'/Users/jakubkostial/Documents/phd/code/ff_project/matrix_generation_dmd/repo/dmd_matrix_gen/masks/aperture1st.npy', large_matrix)




# slm pitch: 8.0 um  ////  8* (54 * 4 = 216) = 1728 um ////1920 x 1080
# dmd pitch: 10.8 um //// 10.8* (40 * 4 = 160) = 1728 um  ////1280 x 800  

# dmd
macropixel_width = 160
macropixel_seperation = 80

# slm
macropixel_width = 216
macropixel_seperation = 108

