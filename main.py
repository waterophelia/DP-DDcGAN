from __future__ import print_function

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage

BATCH_SIZE = 24
EPOCHS = 1
LOGGING = 40
MODEL_SAVE_PATH = '/content/drive/MyDrive/stormy/model/'  # Save path on Google Drive
IMAGE_SAVE_PATH = '/content/drive/MyDrive/stormy/generated_images/'  # Path for saving images in Google Drive
IS_TRAINING = False  # Set this to True for training and False for generating images

def main():
    if IS_TRAINING:
        print('\nBegin to train the network ...\n')
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # Ensure save path directory exists
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)  # Ensure image save path directory exists

        # Load training data
        with h5py.File('Training_Dataset.h5', 'r') as f:
            sources = f['data'][:]

        # Start training
        train(sources, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, logging_period=LOGGING, image_save_period=100, image_save_path=IMAGE_SAVE_PATH)

    else:
        print('\nBegin to generate pictures ...\n')
        path = './test_imgs/'  # Path where test IR and VIS images are stored
        savepath = IMAGE_SAVE_PATH  # Path where generated images will be saved

        # Generate pictures
        Time = []
        for i in range(20):  # Assuming there are 20 image pairs to generate
            index = i + 1
            ir_path = os.path.join(path, f'IR{index}_ds.bmp')
            vis_path = os.path.join(path, f'VIS{index}.bmp')

            begin = time.time()
            model_path = MODEL_SAVE_PATH + 'final_model.ckpt'  # Load the trained model
            generate(ir_path, vis_path, model_path, index, output_path=savepath)
            end = time.time()

            Time.append(end - begin)
            print(f"pic_num: {index}")

        print(f"Time: mean: {np.mean(Time)}, std: {np.std(Time)}")

if __name__ == '__main__':
    main()
