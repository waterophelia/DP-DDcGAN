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
results_dir = './generated_images/'  # Path to your generated images (should be .bmp)
test_imgs_dir = './test_imgs/'  # Path to your test visible and infrared images (should be .bmp)

# Helper function for reading images
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img / 255.0  # Normalize the image to [0, 1]

# Metric Functions
def entropy(img):
    return shannon_entropy(img)

def mean_gradient(img):
    gx, gy = np.gradient(img)
    return np.mean(np.sqrt(gx**2 + gy**2))

def std_deviation(img):
    return np.std(img)

def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=1)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1)

def calculate_correlation_coefficient(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def calculate_spatial_frequency(image):
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return np.sqrt(np.mean(dx**2) + np.mean(dy**2))

# Function to evaluate a single pair of images
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

# Example of applying the metrics over multiple images
def evaluate_dataset(results_dir, test_imgs_dir):
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.bmp')])  # Adjusted to .bmp

    metrics_summary = []
    inference_times = []

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
