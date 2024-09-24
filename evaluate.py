# This code evaluates the quality of fused images generated from visible and infrared images using multiple metrics, 
# such as SSIM, PSNR, entropy, spatial frequency, and correlation coefficients. It also calculates the inference time for each image.

import numpy as np
import time
import os
from skimage import io, img_as_float
import cv2 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy
from skimage.transform import resize

# Paths for results and test images
results_dir = './generated_images/'  # Path to the generated images 
test_imgs_dir = './test_imgs/'  # Path to the original (test) visible and infrared images

# Helper function to load and normalize an image
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img / 255.0

# Helper function for converting image back to uint8
def im2uint8(image):
    return (image * 255).astype(np.uint8)

# Metric Functions

# Calculate the Shannon entropy of an image
def entropy(img):
    return shannon_entropy(img)

# Calculate the mean gradient of an image, representing the sharpness of the image
def mean_gradient(img):
    gx, gy = np.gradient(img)
    return np.mean(np.sqrt(gx**2 + gy**2))

# Calculate the standard deviation of pixel intensities in an image
def std_deviation(img):
    return np.std(img)

# Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
def calculate_psnr(img1, img2):
    img1_uint8 = im2uint8(img1)
    img2_uint8 = im2uint8(img2)
    return psnr(img1_uint8, img2_uint8, data_range=255)

# Calculate Structural Similarity Index (SSIM) between two images
def calculate_ssim(img1, img2):
    img1_uint8 = im2uint8(img1)
    img2_uint8 = im2uint8(img2)
    return ssim(img1_uint8, img2_uint8, data_range=255)

# Calculate the correlation coefficient between two images
def calculate_correlation_coefficient(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

# Calculate the spatial frequency of an image, representing the texture complexity
def calculate_spatial_frequency(image):
    dx = np.diff(image, axis=1) # Horizontal gradient
    dy = np.diff(image, axis=0) # Vertical gradient
    return np.sqrt(np.mean(dx**2) + np.mean(dy**2))

# Function to evaluate a single set of fused, visible, and infrared images
def evaluate_images(fused_img, vis_img, ir_img):
    metrics = {
        "Entropy": entropy(fused_img),
        "Mean Gradient": mean_gradient(fused_img),
        "Standard Deviation": std_deviation(fused_img),
        "Spatial Frequency": calculate_spatial_frequency(fused_img),
        "SSIM with Visible": calculate_ssim(fused_img, vis_img),
        "SSIM with Infrared": calculate_ssim(fused_img, ir_img),
        "PSNR with Visible": calculate_psnr(fused_img, vis_img),
        "PSNR with Infrared": calculate_psnr(fused_img, ir_img),
        "Correlation Coefficient with Visible": calculate_correlation_coefficient(fused_img, vis_img),
        "Correlation Coefficient with Infrared": calculate_correlation_coefficient(fused_img, ir_img)
    }
    return metrics

# Function to evaluate metrics over multiple images in the dataset
def evaluate_dataset(results_dir, test_imgs_dir):
    # List all the result files (fused images) in the results directory
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.bmp')])  # Adjusted to .bmp

    metrics_summary = []  # To store metrics for all images
    inference_times = []  # To store inference times for all images

    for result_file in result_files:
        try:
            start_time = time.time()  # Start timing for inference

            # Load the fused image (generated image)
            fused_img = load_image(os.path.join(results_dir, result_file))

            # Load the corresponding visible and infrared images
            index = result_file.split('.')[0]
            vis_file = f"VIS{index}.bmp"
            ir_file = f"IR{index}_ds.bmp"

            vis_img = load_image(os.path.join(test_imgs_dir, vis_file))
            ir_img = load_image(os.path.join(test_imgs_dir, ir_file))

            # Resize infrared image to match fused image shape
            ir_img = resize(ir_img, fused_img.shape, anti_aliasing=True)

            # Evaluate metrics for the current image set
            metrics = evaluate_images(fused_img, vis_img, ir_img)
            metrics_summary.append(metrics)

            # Calculate inference time
            end_time = time.time()
            inference_times.append(end_time - start_time)

            print(f"Processed {result_file} successfully. Inference time: {inference_times[-1]:.4f} seconds")
        
        except Exception as e:
            # Handle errors that occur during processing of a file
            print(f"Error processing {result_file}: {str(e)}")

    # Return all metrics and inference times
    return metrics_summary, inference_times

# Main function to evaluate the entire dataset
if __name__ == "__main__":
    results_dir = './generated_images/'  # Make sure this directory contains your generated .bmp images
    test_imgs_dir = './test_imgs/'  # Make sure this directory contains your visible and infrared .bmp images

    # Evaluate dataset
    metrics_summary, inference_times = evaluate_dataset(results_dir, test_imgs_dir)

    # Print the average of all metrics
    avg_metrics = {key: np.mean([m[key] for m in metrics_summary]) for key in metrics_summary[0].keys()}
    
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        print(f'{k}: {v:.4f}')
    
    print(f"\nAverage Inference Time: {np.mean(inference_times):.4f} seconds")
