import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

# Load the dataset
with h5py.File('Training_Dataset.h5', 'r') as f:
    a = f['data'][:]

# Rearrange the dimensions of the dataset
sources = np.transpose(a, (0, 3, 2, 1))

# Select one image pair for visualization
vis = sources[100, :, :, 0]  # Visible image
ir = sources[100, :, :, 1]   # Infrared image

# Downscale and upscale the infrared image
ir_ds = scipy.ndimage.zoom(ir, 0.25)           # Downscale by factor 0.25
ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order=3)  # Upscale back to original size using cubic interpolation

# Create a 2x2 grid for displaying images
fig = plt.figure(figsize=(8, 8))
V = fig.add_subplot(221)
I = fig.add_subplot(222)
I_ds = fig.add_subplot(223)
I_ds_us = fig.add_subplot(224)

# Display the images
V.imshow(vis, cmap='gray')
V.set_title('Visible Image')
I.imshow(ir, cmap='gray')
I.set_title('Infrared Image')
I_ds.imshow(ir_ds, cmap='gray')
I_ds.set_title('Downscaled Infrared')
I_ds_us.imshow(ir_ds_us, cmap='gray')
I_ds_us.set_title('Upscaled Infrared')

# Show the plot
plt.tight_layout()
plt.show()
