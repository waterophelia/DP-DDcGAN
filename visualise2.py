# This code visualises the visible and infrared images along with their downscaled and upscaled versions

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

# Load the dataset from HDF5 file
with h5py.File('Training_Dataset.h5', 'r') as f:
    # Load the data array from the file
    a = f['data'][:]

# Rearrange the dimensions of the dataset to match the required format
# Original shape: [samples, height, width, channels], new shape: [samples, channels, width, height]
sources = np.transpose(a, (0, 3, 2, 1))

# Select one specific image pair for visualization (sample index 100)
vis = sources[100, :, :, 0]  # Visible image
ir = sources[100, :, :, 1]   # Infrared image

# Downscale and upscale the infrared image
ir_ds = scipy.ndimage.zoom(ir, 0.25)           # Downscale by 0.25
ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order=3)  # Upscale back to original size using cubic interpolation (order=3)

# Create a 2x2 grid to display the images
fig = plt.figure(figsize=(8, 8)) 
V = fig.add_subplot(221)  
I = fig.add_subplot(222)  
I_ds = fig.add_subplot(223)  
I_ds_us = fig.add_subplot(224)  

# Display the images in grayscale
V.imshow(vis, cmap='gray')
V.set_title('Visible Image')  

I.imshow(ir, cmap='gray')
I.set_title('Infrared Image') 

I_ds.imshow(ir_ds, cmap='gray')
I_ds.set_title('Downscaled Infrared')  

I_ds_us.imshow(ir_ds_us, cmap='gray')
I_ds_us.set_title('Upscaled Infrared')  

plt.tight_layout()

plt.show()
