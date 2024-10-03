from __future__ import print_function

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage

# Hyperparameters
BATCH_SIZE = 24
EPOCHS = 1
LOGGING = 40
MODEL_SAVE_PATH = './model/'  # Path where model checkpoints will be saved
IMAGE_SAVE_PATH = './generated_images/'  # Path where generated images will be saved
IS_TRAINING = True  # True for training, False for generating images
l2_norm_clip = 1.0  # L2 norm bound for DP-SGD
noise_multiplier = 1.1  # Noise multiplier for DP-SGD

def main():
    if IS_TRAINING:
        print('\nBegin to train the network ...\n')
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # Ensure save path directory exists
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)  # Ensure image save path directory exists

        # Load training data
        with h5py.File('Training_Dataset.h5', 'r') as f:
            sources = f['data'][:]

        # Start training with DP-SGD
        train(sources, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, logging_period=LOGGING)

    else:
        print('\nBegin to generate pictures ...\n')
        path = './test_imgs/'
        savepath = './generated_images/'

        # Generate pictures
        Time = []
        for i in range(20):
            index = i + 1
            ir_path = os.path.join(path, f'IR{index}_ds.bmp')
            vis_path = os.path.join(path, f'VIS{index}.bmp')

            begin = time.time()
            model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.ckpt')
            generate(ir_path, vis_path, model_path, index=index, output_path=savepath)
            end = time.time()

            Time.append(end - begin)
            print(f"pic_num: {index}")

        print(f"Time: mean: {np.mean(Time)}, std: {np.std(Time)}")

if __name__ == '__main__':
    main()
