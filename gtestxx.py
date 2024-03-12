import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Gaussian distribution parameters and recalculations
mu = 0  # Mean
sigma = 1  # Standard deviation
x = np.linspace(-5, 5, 1000)
pdf_values = norm.pdf(x, mu, sigma)
scaled_pdf_values = (pdf_values - min(pdf_values)) / (max(pdf_values) - min(pdf_values))
threshold = 0.5
rounded_pdf_values = np.where(scaled_pdf_values >= threshold, 1, 0)

n_rows = 1080 # Number of rows in the original matrix
n_cols = len(x)  # Number of columns in the original matrix

# Re-initialize the original matrix with rounded Gaussian distribution values
matrix = np.zeros((n_rows, n_cols))
for i in range(n_rows):
    matrix[i, :] = rounded_pdf_values

# Define the number of times N the smaller matrix should be replicated and spacing
N = 4  # Example value
spacing_columns = 5  # Example value for additional spacing

# Adjusted calculation for the required size of the larger matrix with column-wise placement and spacing
larger_matrix_cols = (n_cols + spacing_columns) * N - spacing_columns
larger_matrix_rows = n_rows

# Initialize the larger matrix with spacing for column-wise placement
larger_matrix_col_wise_with_spacing = np.zeros((larger_matrix_rows, larger_matrix_cols))

# Copy the smaller matrix into the larger matrix N times with spacing
for i in range(N):
    start_col = i * (n_cols + spacing_columns)
    larger_matrix_col_wise_with_spacing[:, start_col:start_col + n_cols] = matrix


# Verify by displaying a portion of the larger matrix










plt.figure(figsize=(8, 4))
plt.imshow(larger_matrix_col_wise_with_spacing, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()
