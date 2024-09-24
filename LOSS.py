import tensorflow as tf
import numpy as np

# SSIM loss function to compute the Structural Similarity Index between two images
def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    # Generate a Gaussian window for SSIM calculation
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    
    # SSIM parameters (K1, K2) and dynamic range (L)
    K1 = 0.01  # SSIM constant to stabilize division
    K2 = 0.03  # SSIM constant to stabilize division
    L = 1      # Image depth (1 when images are normalized to [0, 1], 255 for 8-bit images)
    C1 = (K1 * L) ** 2  # Constant based on L and K1
    C2 = (K2 * L) ** 2  # Constant based on L and K2

    # Calculate the means (mu1, mu2) of both images using a Gaussian filter (convolution)
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')

    # Square of the means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    # Product of the means
    mu1_mu2 = mu1 * mu2

    # Calculate variances (sigma1_sq, sigma2_sq) and covariance (sigma12)
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    # SSIM calculation: compute the similarity measure
    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # Reduce to the mean value across all pixels to get the final SSIM loss
    value = tf.reduce_mean(value)
    
    return value

# L1 loss function, which calculates the sum of absolute differences (L1 norm)
def L1_LOSS(batchimg):
    # Calculate the L1 norm (sum of absolute values) across height and width dimensions
    L1_norm = tf.reduce_sum(tf.abs(batchimg), axis=[1, 2])
    
    # Compute the mean L1 loss over the batch
    E = tf.reduce_mean(L1_norm)
    
    return E

# Frobenius loss function, which computes the Frobenius norm (L2 norm) over images
def Fro_LOSS(batchimg):
    # Calculate Frobenius norm 
    fro_norm = tf.square(tf.norm(batchimg, axis=[1, 2], ord='fro'))
    
    # Compute the mean Frobenius loss over the batch
    E = tf.reduce_mean(fro_norm)
    
    return E

# Function to generate a Gaussian kernel, mimicking the 'fspecial' Gaussian function in MATLAB
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' Gaussian MATLAB function"""
    
    # Generate a 2D grid of x and y coordinates centered at zero
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    
    # Expand dimensions to make x_data and y_data compatible with TensorFlow conv2d operation
    x_data = np.expand_dims(x_data, axis=-1)  # Shape becomes [size, size, 1]
    x_data = np.expand_dims(x_data, axis=-1)  # Shape becomes [size, size, 1, 1]
    y_data = np.expand_dims(y_data, axis=-1)  # Shape becomes [size, size, 1]
    y_data = np.expand_dims(y_data, axis=-1)  # Shape becomes [size, size, 1, 1]

    # Convert to TensorFlow tensors
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    # Create a 2D Gaussian window 
    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    # Normalize the Gaussian kernel so the sum of all elements equals 1
    g_sum = tf.reduce_sum(g)
    
    return g / g_sum  # Return the normalized Gaussian kernel
