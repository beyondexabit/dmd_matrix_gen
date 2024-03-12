import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm


# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

def generate_intensity_pattern(input_matrix, macropixel_width, macropixel_seperation, scaling_factor):
    """
    Generate a phase pattern for an LC-SLM based on the given matrix.
    Args:
    - matrix (2D numpy array): The matrix to be encoded, with values from -1 to 1.
    - macropixel_size (int): The size of each macropixel (n x n pixels).
    - grating_frequency_x (float): Frequency of the grating pattern within each macropixel on the x-axis.
    - grating_frequency_y (float): Frequency of the grating pattern within each macropixel on the y-axis.
    Returns:
    - 2D numpy array: The generated phase pattern.
    """
    threshold = 0.5  # This is an arbitrary choice for demonstration; adjust as needed
    elements = input_matrix[0]

    n_rows = 800  # Number of rows in the matrix
    n_cols = macropixel_width  # Number of columns in the matrix, matching the distribution resolution

    pattern = np.zeros((n_rows, len(elements) * macropixel_width + (len(elements) - 1) * macropixel_seperation))
    x = np.linspace(0, macropixel_width, macropixel_width)

    count = 0
    for i in elements:
        amplitude = np.abs(i)*scaling_factor
        pdf_values = norm.pdf(x, int(macropixel_width/2), amplitude)
        scaled_pdf_values = (pdf_values - min(pdf_values)) / (max(pdf_values) - min(pdf_values))
        rounded_pdf_values = np.where(scaled_pdf_values >= threshold, 1, 0)

        # Initialize an empty matrix
        grating_combined = np.zeros((n_rows, n_cols))

        # Fill each row of the matrix with the rounded Gaussian distribution values
        for i in range(n_rows):
            grating_combined[i, :] = rounded_pdf_values
        
        start_col = count * (n_cols + macropixel_seperation)
        # Place the macropixel in the pattern
        pattern[:, start_col:start_col + n_cols] = grating_combined
        count += 1

    return pattern


# Example usage

# make sure to only input 1d array
input_matrix = np.array([[1, 1]])
macropixel_width = 160
macropixel_seperation = 80
scaling_factor = 100

matrix_new = generate_intensity_pattern(input_matrix, macropixel_width, macropixel_seperation, scaling_factor)


# Rescale matrix between 0 and 128

# Create a 1920x1080 matrix initialized to 0
large_matrix = np.zeros((800, 1200))

# Calculate center coordinates
center_x = (large_matrix.shape[1] - matrix_new.shape[1]) // 2
center_y = (large_matrix.shape[0] - matrix_new.shape[0]) // 2

# Place the rescaled matrix in the center of the larger matrix
large_matrix[center_y:center_y + matrix_new.shape[0], center_x:center_x + matrix_new.shape[1]] = matrix_new

plt.figure()
plt.imshow(large_matrix, cmap='viridis', interpolation='nearest')
plt.title('Transform')
plt.show()

large_matrix = rescale_matrix(large_matrix, 0, 255)


np.save(f'/Users/jakubkostial/Documents/phd/code/ff_project/matrix_generation_dmd/repo/dmd_matrix_gen/masks/testgrating_ti.npy', large_matrix)


# slm pitch: 8.0 um  ////  8* (54 * 4 = 216) = 1728 um ////1920 x 1080
# dmd pitch: 10.8 um //// 10.8* (40 * 4 = 160) = 1728 um  ////1280 x 800  

# dmd
macropixel_width = 160
macropixel_seperation = 80

# slm
macropixel_width = 216
macropixel_seperation = 108

