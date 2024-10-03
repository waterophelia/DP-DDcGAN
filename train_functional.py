# This code implements the training loop for the DDcGAN. It includes the Generator and two Discriminators, 
# and trains them using various loss functions.

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from datetime import datetime
import os

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from generate import generate

# Hyperparameters
patch_size = 84
LEARNING_RATE = 0.0002
DECAY_RATE = 0.9
eps = 1e-8
rc = 4 # Rescaling factor for some layers

# Function to compute the gradient of the image using a custom kernel
def compute_gradient(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g

# Function to add noise to the coefficients of loss terms (functional mechanism)
def add_noise_to_coefficient(coefficient, sensitivity, epsilon):
    #noise = np.random.laplace(0, sensitivity / epsilon) # For Laplace noise
    noise = np.random.normal(0, sensitivity / epsilon) # For Gaussian noise
    return coefficient + noise

# Main training function
def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period=10):
    start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print('Epochs: %d, Batch size: %d' % (EPOCHS, BATCH_SIZE))
    
    checkpoint_save_path = save_path + 'temporary.ckpt'
    num_imgs = source_imgs.shape[0]  # Number of training images
    mod = num_imgs % BATCH_SIZE  # Check if there is any leftover data that doesn't fit into a full batch
    n_batches = int(num_imgs // BATCH_SIZE)  # Number of batches
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    # Trim the dataset if there are leftover images
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source_imgs = source_imgs[:-mod]

    # Create the models
    G = Generator('Generator')
    D1 = Discriminator1('Discriminator1')
    D2 = Discriminator2('Discriminator2')

    current_iter = tf.Variable(0, trainable=False)
    # Learning rate schedule with exponential decay
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=int(n_batches), decay_rate=DECAY_RATE, staircase=False)

    # Optimizers
    G_GAN_solver = tf.keras.optimizers.RMSprop(learning_rate)
    G_solver = tf.keras.optimizers.RMSprop(learning_rate)
    D1_solver = tf.keras.optimizers.SGD(learning_rate)
    D2_solver = tf.keras.optimizers.SGD(learning_rate)

    # Set up checkpoints
    checkpoint = tf.train.Checkpoint(G=G, D1=D1, D2=D2, G_GAN_solver=G_GAN_solver, G_solver=G_solver, 
                                     D1_solver=D1_solver, D2_solver=D2_solver, current_iter=current_iter)
    manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=500)

    # Training function for the Generator
    @tf.function
    def train_G(VIS_batch, ir_batch, sensitivity=1.0, epsilon=1):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # Downsample ir_batch to match the dimensions of generated_img_ds
            ir_batch_ds = tf.nn.avg_pool(ir_batch, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            # Compute difference and inspect
            diff = generated_img_ds - ir_batch_ds
            grad_of_vis = compute_gradient(VIS_batch)  # Using the grad function defined above

            # Discriminator outputs for the generated image
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)

            # GAN loss for the generator
            G_loss_GAN_D1 = -tf.reduce_mean(tf.math.log(D1_fake + eps))
            G_loss_GAN_D2 = -tf.reduce_mean(tf.math.log(D2_fake + eps))

            # Apply functional mechanism noise to the GAN loss coefficients
            lambda_gan_d1 = add_noise_to_coefficient(1.3, sensitivity, epsilon)
            lambda_gan_d2 = add_noise_to_coefficient(0.7, sensitivity, epsilon)
            G_loss_GAN = lambda_gan_d1 * G_loss_GAN_D1 + lambda_gan_d2 * G_loss_GAN_D2

            # Additional losses for the generator 
            LOSS_IR = Fro_LOSS(diff)
            LOSS_VIS = L1_LOSS(compute_gradient(generated_img) - grad_of_vis)
            SSIM_loss_VIS = SSIM_LOSS(generated_img, VIS_batch)  # SSIM with visible image
            SSIM_loss_IR = SSIM_LOSS(generated_img, ir_batch)    # SSIM with infrared image

            lambda_ir = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            lambda_vis = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            G_loss_norm = lambda_ir * LOSS_IR + lambda_vis * LOSS_VIS + 2.0 * (SSIM_loss_IR + SSIM_loss_VIS)

            G_loss = G_loss_GAN + 0.8 * G_loss_norm

        # Compute and apply gradients
        gradients = tape.gradient(G_loss, G.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        G_solver.apply_gradients(zip(clipped_gradients, G.trainable_variables))
        
        return G_loss, G_loss_GAN_D1, G_loss_GAN_D2, D1_fake, D2_fake, generated_img

    # Training function for Discriminator1 
    @tf.function
    def train_D1(VIS_batch, ir_batch, sensitivity=1.0, epsilon=1):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)

            # Pass through the discriminator
            D1_real = D1(VIS_batch, training=True)
            D1_fake = D1(generated_img, training=True)

            # GAN loss for the discriminator
            D1_loss_real = -tf.reduce_mean(tf.math.log(D1_real + eps))
            D1_loss_fake = -tf.reduce_mean(tf.math.log(1. - D1_fake + eps))
            
            lambda_real = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            lambda_fake = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            D1_loss = lambda_real * D1_loss_real + lambda_fake * D1_loss_fake

        # Compute and apply gradients
        gradients = tape.gradient(D1_loss, D1.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        D1_solver.apply_gradients(zip(clipped_gradients, D1.trainable_variables))
        
        return D1_loss, D1_real, D1_fake

    # Training function for Discriminator2 
    @tf.function
    def train_D2(VIS_batch, ir_batch, sensitivity=1.0, epsilon=1):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            generated_img_ds = g0  # Output shape should be [batch_size, 21, 21, channels]

            # Pass through the discriminator
            D2_real = D2(ir_batch, training=True)
            D2_fake = D2(generated_img_ds, training=True)

            # GAN loss for the discriminator
            D2_loss_real = -tf.reduce_mean(tf.math.log(D2_real + eps))
            D2_loss_fake = -tf.reduce_mean(tf.math.log(1. - D2_fake + eps))
            
            lambda_real = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            lambda_fake = add_noise_to_coefficient(0.5, sensitivity, epsilon)
            D2_loss = lambda_real * D2_loss_real + lambda_fake * D2_loss_fake

        # Compute and apply gradients
        gradients = tape.gradient(D2_loss, D2.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        D2_solver.apply_gradients(zip(clipped_gradients, D2.trainable_variables))
        
        return D2_loss, D2_real, D2_fake

    # Function for training the generator with respect to the GAN loss
    @tf.function
    def train_G_GAN(VIS_batch, ir_batch, sensitivity=1.0, epsilon=1):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            # Downsample the generated image twice
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Feed the generated image to the discriminators D1 and D2
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)

            # Compute the GAN loss for fooling D1 and D2
            G_loss_GAN_D1 = -tf.reduce_mean(tf.math.log(D1_fake + eps))
            G_loss_GAN_D2 = -tf.reduce_mean(tf.math.log(D2_fake + eps))
            
            # Apply functional mechanism noise to the GAN loss coefficients
            lambda_gan_d1 = add_noise_to_coefficient(1.0, sensitivity, epsilon)
            lambda_gan_d2 = add_noise_to_coefficient(1.0, sensitivity, epsilon)
            G_loss_GAN = lambda_gan_d1 * G_loss_GAN_D1 + lambda_gan_d2 * G_loss_GAN_D2

        # Compute gradients for generator based on the GAN loss
        gradients = tape.gradient(G_loss_GAN, G.trainable_variables)
        # Clip gradients to avoid exploding gradients
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        G_GAN_solver.apply_gradients(zip(clipped_gradients, G.trainable_variables))
        
        return G_loss_GAN

    # Start Training 
    step = 0
    count_loss = 0
    num_imgs = source_imgs.shape[0]

    for epoch in range(EPOCHS):
        np.random.shuffle(source_imgs) # Shuffle the dataset at the beginning of each epoch
        for batch in range(n_batches):
          step += 1
          current_iter.assign(step)

           # Extract batches of visible and infrared images
          VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
          ir_or_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
          
          # Resize both the visible and infrared patches to match the patch size
          VIS_batch = np.array([ndimage.zoom(vis_patch, (patch_size / vis_patch.shape[0], patch_size / vis_patch.shape[1])) for vis_patch in VIS_batch])
          ir_batch = np.array([ndimage.zoom(ir_patch, (patch_size / ir_patch.shape[0], patch_size / ir_patch.shape[1])) for ir_patch in ir_or_batch])

          # Expand dimensions to match the expected input format for the model
          VIS_batch = np.expand_dims(VIS_batch, -1)
          ir_batch = np.expand_dims(ir_batch, -1)

          # Convert to TensorFlow tensors
          VIS_batch = tf.convert_to_tensor(VIS_batch, dtype=tf.float32)
          ir_batch = tf.convert_to_tensor(ir_batch, dtype=tf.float32)

          it_G = 0
          it_D1 = 0
          it_D2 = 0

          # Alternating training: Discriminators on even batches, Generator on odd batches
          if batch % 2 == 0:
                D1_loss, D1_real, D1_fake = train_D1(VIS_batch, ir_batch)
                it_D1 += 1
                D2_loss, D2_real, D2_fake = train_D2(VIS_batch, ir_batch)
                it_D2 += 1
                G_loss = 0  # Placeholder value
                G_loss_GAN_D1 = 0  # Placeholder value
                G_loss_GAN_D2 = 0  # Placeholder value
          else:
              # Call train_G and get the generated image
              G_loss, G_loss_GAN_D1, G_loss_GAN_d2, D1_fake, D2_fake, generated_img = train_G(VIS_batch, ir_batch)
              it_G += 1
              D1_loss = D2_loss = 0  # Placeholder values

          # Additional logic for re-training discriminators if losses exceed certain thresholds
          if batch % 2 == 0:
              while D1_loss > 1.9 and it_D1 < 20:
                  D1_loss, _, _ = train_D1(VIS_batch, ir_batch)
                  it_D1 += 1
              while D2_loss > 1.9 and it_D2 < 20:
                  D2_loss, _, _ = train_D2(VIS_batch, ir_batch)
                  it_D2 += 1
          else:
              while (D1_loss < 1 or D2_loss < 1) and it_G < 20:
                  G_loss_GAN = train_G_GAN(VIS_batch, ir_batch)
                  G_loss, G_loss_GAN_D1, G_loss_GAN_D2, D1_fake, D2_fake, generated_img = train_G(VIS_batch, ir_batch)
                  it_G += 1
              while G_loss > 200 and it_G < 20:
                  G_loss, _, _, _, _, _ = train_G(VIS_batch, ir_batch)
                  it_G += 1

          # Logging the training process
          if batch % 1 == 0:
                elapsed_time = datetime.now() - start_time
                lr = learning_rate(current_iter)
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch}/{n_batches}, G_loss: {G_loss}, D1_loss: {D1_loss}, D2_loss: {D2_loss}")
                print(f"Learning Rate: {lr}, Elapsed Time: {elapsed_time}\n")

          # Save checkpoints periodically
          if step % logging_period == 0:
                manager.save()

    # Save the model after training
    manager.save()
    # Add this at the end of your training function
    total_elapsed_time = datetime.now() - start_time
    print(f"Total Elapsed Time for Training: {total_elapsed_time}")

    print("Training completed.")
