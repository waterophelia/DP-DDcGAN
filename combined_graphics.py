# This code generates combined images for comparison. You need to have the images already generated in order to use it

from PIL import Image, ImageOps, ImageDraw, ImageFont
import os

# Define the base folder containing all subfolders
base_folder = "/Users/an.balcer/Desktop/generated/"

# List of subfolders to be used as rows (with labels for row headers, split into two lines where needed)
subfolders = [
    ("MRI\nimage", "mri"), 
    ("PET\nimage", "pet"), 
    ("Base\nmodel", "base"), 
    ("Functional\nMechanism\nwith Gaussian\nε = 0.1", "functional-gaussian-0.1"), 
    ("Functional\nMechanism\nwith Gaussian\nε = 1", "functional-gaussian-1"), 
    ("Functional\nMechanism\nwith Gaussian\nε = 5", "functional-gaussian-5")
]

# The specific image indices we want to pick from each folder
image_indices = [2, 5, 8, 13, 15, 19]  # These correspond to the filenames
image_filenames = [f"{index}.png" for index in image_indices]  # Create filenames like '2.bmp', '8.bmp', etc.

min_width, min_height = float('inf'), float('inf')

for label, folder in subfolders:
    for filename in image_filenames:
        img_path = os.path.join(base_folder, folder, filename)
        try:
            img = Image.open(img_path)
            width, height = img.size
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Define white space padding between rows and columns
row_padding = 10  # Padding between rows
col_padding = 10  # Padding between columns
label_width = 150  # Increase space for row labels to accommodate two lines of text

# Create a blank canvas for the grid (6 columns, 6 rows, with padding and space for labels)
new_image_width = 6 * (min_width + col_padding) + label_width
new_image_height = 6 * (min_height + row_padding)
new_image = Image.new('RGB', (new_image_width, new_image_height), (255, 255, 255))

# Provide the full path to the font file
font_path = "/Library/Fonts/Times New Roman.ttf"  
# Font settings for the row labels
font = ImageFont.truetype(font_path, 26)  

draw = ImageDraw.Draw(new_image)

# Add row labels on the left side (subfolder names), with line breaks where needed
for row, (label, folder) in enumerate(subfolders):
    # Split label into lines 
    lines = label.split("\n")
    
    # Calculate total label height 
    total_label_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines]) + ((len(lines) - 1) * 5)
    
    # Center the entire block of text vertically
    label_y = row * (min_height + row_padding) + (min_height // 2) - (total_label_height // 2)
    
    # Draw each line of the label, centered horizontally
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        label_x = (label_width // 2) - (bbox[2] // 2)  
        draw.text((label_x, label_y), line, font=font, fill=(0, 0, 0))
        label_y += bbox[3] + 5  # Move to the next line position

# Loop through the subfolders (rows) and image indices (columns)
for row, (label, folder) in enumerate(subfolders):
    for col, filename in enumerate(image_filenames):
        # Open the corresponding image from the current subfolder and filename
        img_path = os.path.join(base_folder, folder, filename)
        try:
            img = Image.open(img_path)
            # Resize the image to the smallest size found
            img_resized = ImageOps.fit(img, (min_width, min_height), method=Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading or resizing image {img_path}: {e}")
            continue
        
        # Calculate position in the grid (rows are subfolders, columns are images)
        x = col * (min_width + col_padding) + label_width
        y = row * (min_height + row_padding)
        
        # Paste the resized image at the calculated position
        new_image.paste(img_resized, (x, y))

# Save and show the combined image
combined_image_path = os.path.join(base_folder, "func_gaussian.png")
new_image.save(combined_image_path)
new_image.show()
