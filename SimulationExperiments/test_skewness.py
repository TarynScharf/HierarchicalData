import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm

# Set the number of points and the number of iterations
n_points = 1000
n_iterations = 10

# Create the plots
plt.figure(figsize=(12, 8))

# Loop to generate and plot the skewed distributions
for i in range(n_iterations):
    # Generate data with increasing skewness
    skewness = i  # Increase skewness by 1 each time
    data = skewnorm.rvs(a=skewness, size=n_points)

    # Plot the KDE of the sampled data
    plt.subplot(5, 2, i + 1)  # Create a subplot for each distribution
    sns.kdeplot(data, shade=True, color='b', alpha=0.6)  # KDE plot
    plt.title(f'Skewness = {skewness}')

    # Dynamically adjust the x and y limits
    plt.xlim(np.min(data) - 1, np.max(data) + 1)
    plt.ylim(0, np.max(sns.kdeplot(data).get_lines()[0].get_data()[1]) * 1.1)  # Adjust y-axis to fit KDE curve

# Adjust layout
plt.tight_layout()
plt.show()
