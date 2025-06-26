import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the number of points and the number of iterations
n_points = 1000
n_iterations = 5  # We will now plot 5 graphs

# Set the color palette for the KDEs
colors = sns.color_palette("tab10", n_iterations)

# Create the plot
plt.figure(figsize=(8, 6))

# Loop to generate and plot the KDE for data with decreasing scale by 2 (from 10 to 2)
for i, scale in enumerate(range(10, 0, -2)):  # Scale values: 10, 8, 6, 4, 2
    data = np.random.normal(loc=0, scale=scale, size=n_points)

    # Plot the KDE of the sampled data with a different color for each iteration
    sns.kdeplot(data, fill=True, color=colors[i], alpha=0.4, label=f'Scale = {scale}')

# Add labels and title
plt.title('Kernel Density Estimates with Decreasing Variability (Scale)')
plt.xlabel('Value')
plt.ylabel('Density')

# Add legend
plt.legend()

# Show the plot
plt.show()
