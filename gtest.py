import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Parameters for the Gaussian distribution
mu = 0  # Mean
sigma = 1  # Standard deviation

# Define a range of x values
x = np.linspace(-5, 5, 100)

# Calculate the PDF for each x value

# Plotting
threshold = 0.5  # This is an arbitrary choice for demonstration; adjust as needed
# Rounding the PDF values to 0 or 1 based on the threshold

plt.figure(figsize=(8, 4))
for i in np.linspace(0, 1, 5):
    pdf_values = norm.pdf(x, mu, i)
    scaled_pdf_values = (pdf_values - min(pdf_values)) / (max(pdf_values) - min(pdf_values))
    # Rounding the PDF values to 0 or 1 based on the threshold
    rounded_pdf_values = np.where(scaled_pdf_values >= threshold, 1, 0)
    plt.plot(x, rounded_pdf_values, label='Gaussian Distribution\n$\mu=0$, $\sigma=1$')

plt.title('Gaussian Distribution PDF')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.show()