# This code loads and visualizes a few random samples (visible and infrared images) from a dataset stored in an HDF5 file.
# The selected samples are displayed and saved as an image file for easy inspection.

import numpy as np
import matplotlib.pyplot as plt
import h5py
from IPython.display import Image, display

def visualize_and_save_dataset_samples(dataset_path, num_samples=5):
    # Load the dataset
    with h5py.File(dataset_path, 'r') as f:
        a = f['data'][:]

    # Rearrange the dimensions of the dataset
    sources = np.transpose(a, (0, 3, 2, 1))
    
    # Set up the plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    
    for i in range(num_samples):
        # Select random index
        index = np.random.randint(0, sources.shape[0])
        
        # Extract visible and infrared patches
        vis_patch = sources[index, :, :, 0]  # Visible image
        ir_patch = sources[index, :, :, 1]   # Infrared image
        
        # Plot the patches
        axes[i, 0].imshow(vis_patch, cmap='gray')
        axes[i, 0].set_title(f'Visible Patch {index}')
        axes[i, 1].imshow(ir_patch, cmap='gray')
        axes[i, 1].set_title(f'Infrared Patch {index}')
        
        # Remove axes for clarity
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    
    # Save the figure
    fig.savefig('dataset_samples_visualization.png')
    plt.close(fig)  # Close the plot to avoid inline display

    # Display the saved image
    display(Image('dataset_samples_visualization.png'))

# Call the function to visualize and save dataset samples
visualize_and_save_dataset_samples('Training_Dataset.h5', num_samples=5)
