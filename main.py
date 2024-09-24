# This code manages the training and generation of images using a pre-trained model. It can either train the model 
# using a dataset or generate fused images from infrared and visible input images. It includes functionality for logging, 
# saving model checkpoints, and saving generated images.

from __future__ import print_function

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage

# Hyperparameters and settings
BATCH_SIZE = 24
EPOCHS = 1
LOGGING = 40
MODEL_SAVE_PATH = './model/' # Path where the trained model will be saved
IMAGE_SAVE_PATH = './generated_images/' # Path where generated images will be saved
IS_TRAINING = True # True for model training and False for generating

def main():
    if IS_TRAINING:
        print('\nBegin to train the network ...\n')
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # Ensure model save path directory exists
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)  # Ensure image save path directory exists

        # Load training data
        with h5py.File('Training_Dataset.h5', 'r') as f:
            sources = f['data'][:]

        # Start training
        train(sources, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, logging_period=LOGGING, image_save_period=100, image_save_path=IMAGE_SAVE_PATH)

    else:
        print('\nBegin to generate pictures ...\n')
        path = './test_imgs/' # Directory containing test images
        savepath = './generated_images/'

        Time = [] # List to store the time taken for each generation step
        for i in range(20):
            index = i + 1
            ir_path = os.path.join(path, f'IR{index}_ds.bmp')
            vis_path = os.path.join(path, f'VIS{index}.bmp')

            begin = time.time()
            model_path = MODEL_SAVE_PATH + 'final_model.ckpt'
            generate(ir_path, vis_path, model_path, index, output_path=savepath)
            end = time.time()

            Time.append(end - begin)
            print(f"pic_num: {index}")

         # Calculate and print the average and standard deviation of generation times
        print(f"Time: mean: {np.mean(Time)}, std: {np.std(Time)}")

if __name__ == '__main__':
    main()
