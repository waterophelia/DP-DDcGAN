import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths for results and test images
results_dir = './generated_images/'  
test_imgs_dir = './test_imgs/'  

# Load image helper
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img / 255.0  # Normalize the image to [0, 1]

# Function to create and save side-by-side comparison
def create_comparison_image(fused_img, vis_img, ir_img, save_path, index):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display visible image
    axs[0].imshow(vis_img, cmap='gray')
    axs[0].set_title('Visible Image')
    axs[0].axis('off')

    # Display infrared image
    axs[1].imshow(ir_img, cmap='gray')
    axs[1].set_title('Infrared Image')
    axs[1].axis('off')

    # Display fused image
    axs[2].imshow(fused_img, cmap='gray')
    axs[2].set_title('Fused Image')
    axs[2].axis('off')

    # Save the comparison image
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"comparison_{index}.png"), bbox_inches='tight')
    plt.close()

    print(f"Comparison image saved for {index}")

# Create comparison images for multiple fused images
def generate_comparison_images(results_dir, test_imgs_dir, save_path):
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.bmp')])
    
    for result_file in result_files:
        try:
            # Load the fused image (generated image)
            fused_img = load_image(os.path.join(results_dir, result_file))

            # Load the corresponding visible and infrared images
            index = result_file.split('.')[0]
            vis_file = f"VIS{index}.bmp"
            ir_file = f"IR{index}_ds.bmp"

            vis_img = load_image(os.path.join(test_imgs_dir, vis_file))
            ir_img = load_image(os.path.join(test_imgs_dir, ir_file))

            # Create the comparison image
            create_comparison_image(fused_img, vis_img, ir_img, save_path, index)
        
        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")

# Main function to run the image comparison generation
if __name__ == "__main__":
    results_dir = './generated_images/'
    test_imgs_dir = './test_imgs/'
    save_path = './comparisons/'  # Folder to save comparison images

    # Generate comparison images
    generate_comparison_images(results_dir, test_imgs_dir, save_path)
