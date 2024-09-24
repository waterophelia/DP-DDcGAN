# This code loads generated fused images along with their corresponding visible and infrared test images,
# and generates side-by-side comparison images for visualization. It saves the comparisons as image files.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths for the directories containing results and test images
results_dir = './generated_images/'  # where the generated fused images are stored
test_imgs_dir = './test_imgs/'       # where the original visible and infrared test images are stored

# Helper function to load an image, convert it to grayscale, and normalize it to the range [0, 1]
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    return img / 255.0  

# Function to create and save a side-by-side comparison of visible, infrared, and fused images
def create_comparison_image(fused_img, vis_img, ir_img, save_path, index):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(vis_img, cmap='gray')
    axs[0].set_title('Visible Image')  
    axs[0].axis('off')  

    axs[1].imshow(ir_img, cmap='gray')
    axs[1].set_title('Infrared Image')  
    axs[1].axis('off')  

    axs[2].imshow(fused_img, cmap='gray')
    axs[2].set_title('Fused Image')  
    axs[2].axis('off') 

    # Ensure the directory exists and save the comparison image
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"comparison_{index}.png"), bbox_inches='tight')  
    plt.close()  

    print(f"Comparison image saved for {index}")  

# Function to generate and save comparison images for multiple fused images
def generate_comparison_images(results_dir, test_imgs_dir, save_path):
    # List all the result files (fused images) from the results directory
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.bmp')])

    # Iterate over all the result files
    for result_file in result_files:
        try:
            # Load the fused image (generated image) from the results directory
            fused_img = load_image(os.path.join(results_dir, result_file))

            # Extract the image index from the result file name (e.g., "001.bmp" -> index "001")
            index = result_file.split('.')[0]

            # Construct the file names for the corresponding visible and infrared test images
            vis_file = f"VIS{index}.bmp"  
            ir_file = f"IR{index}_ds.bmp"  

            # Load the visible and infrared images
            vis_img = load_image(os.path.join(test_imgs_dir, vis_file))
            ir_img = load_image(os.path.join(test_imgs_dir, ir_file))

            # Create and save the comparison image
            create_comparison_image(fused_img, vis_img, ir_img, save_path, index)
        
        except Exception as e:
            # Handle any errors that occur during processing of a file
            print(f"Error processing {result_file}: {str(e)}")

# Main function to run the image comparison generation
if __name__ == "__main__":
    results_dir = './generated_images/'  # Directory containing fused images
    test_imgs_dir = './test_imgs/'       # Directory containing visible and infrared images
    save_path = './comparisons/'         # Directory where comparison images will be saved

    # Generate comparison images for all images in the results directory
    generate_comparison_images(results_dir, test_imgs_dir, save_path)
