import tensorflow as tf
import numpy as np

def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    # Generate a Gaussian window for SSIM calculation
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    
    # SSIM parameters
    K1 = 0.01
    K2 = 0.03
    L = 1  # Depth of image (255 in case the image has a different scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate means (mu1, mu2)
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    #tf.print("SSIM_LOSS - mu1 shape:", tf.shape(mu1), "mu2 shape:", tf.shape(mu2))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate variances (sigma1, sigma2) and covariance (sigma12)
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    #tf.print("SSIM_LOSS - sigma1_sq shape:", tf.shape(sigma1_sq), 
    #         "sigma2_sq shape:", tf.shape(sigma2_sq), 
    #         "sigma12 shape:", tf.shape(sigma12))

    # SSIM calculation
    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    value = tf.reduce_mean(value)
    #tf.print("SSIM_LOSS - final value:", value)
    return value

def L1_LOSS(batchimg):
    # Calculate the L1 norm
    L1_norm = tf.reduce_sum(tf.abs(batchimg), axis=[1, 2])
    #tf.print("L1_LOSS - L1_norm shape:", tf.shape(L1_norm))
    E = tf.reduce_mean(L1_norm)
    #tf.print("L1_LOSS - final value:", E)
    return E

def Fro_LOSS(batchimg):
    # Calculate Frobenius norm
    fro_norm = tf.square(tf.norm(batchimg, axis=[1, 2], ord='fro'))
    #tf.print("Fro_LOSS - fro_norm shape:", tf.shape(fro_norm))
    E = tf.reduce_mean(fro_norm)
    #tf.print("Fro_LOSS - final value:", E)
    return E

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    
    # Expand dimensions to fit the expected input shape for conv2d
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    # Create a Gaussian window
    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g_sum = tf.reduce_sum(g)
    #tf.print("Gaussian kernel shape:", tf.shape(g))
    #tf.print("Sum of Gaussian kernel elements (should be 1 after normalization):", g_sum)
    
    return g / g_sum
