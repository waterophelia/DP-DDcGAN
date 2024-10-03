# This code generates the graphs for privacy-utility trade-offs. Here, we have the data for Laplace inputs.

import matplotlib.pyplot as plt
import seaborn as sns

# Data from the table for the Laplace mechanism for outputs
metrics = ['EN (bits)', 'MG', 'SD', 'PSNR (dB)', 'SSIM', 'SF', 'Training\nTime (min)']
privacy_levels = ['Base Model (no $\epsilon$)', r'$\epsilon = 0.1$', r'$\epsilon = 1$', r'$\epsilon = 5$']
data = [
    [7.4075, 7.3744, 7.4606, 7.4447],  # EN
    [0.1457, 0.1409, 0.1539, 0.1581],  # MG
    [0.1682, 0.1670, 0.1762, 0.1787],  # SD
    [(10.6023 + 11.5141) / 2, (10.2958 + 11.9577) / 2, (10.5223 + 11.0277) / 2, (10.4277 + 10.8052) / 2],  # PSNR (combined vis and inf)
    [(0.0407 + 0.0331) / 2, (0.0417 + 0.0381) / 2, (0.0298 + 0.0286) / 2, (0.0300 + 0.0266) / 2],  # SSIM (combined vis and inf)
    [0.3237, 0.3230, 0.3501, 0.3511],  # SF
    [71.5973, 71.8085, 71.6162, 71.4706],  # Training Time (min)
]

color = 'green'

# Create subplots
fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 7), sharex=True)

for i, metric in enumerate(metrics):
    axes[i].plot(privacy_levels, data[i], marker='.', color=color)
    axes[i].set_ylabel(metric, labelpad=20)  # Justify y-labels to the left
    axes[i].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position further left
    axes[i].grid(True)

# Set title and overall labels
axes[0].set_title('Input - Laplace - Utility Metrics vs Privacy')
plt.xlabel('Privacy Levels')
plt.tight_layout()

# Show plot
plt.show()
