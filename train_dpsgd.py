import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from datetime import datetime
import os
from scipy import ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS


# Training parameters
patch_size = 84
LEARNING_RATE = 0.0002
DECAY_RATE = 0.9
l2_norm_clip = 1.0
noise_multiplier = 1.1
eps = 1e-8

# Function to compute gradients for an image
def compute_gradient(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g

# DP-SGD optimizer creation
def create_dp_optimizer(learning_rate, l2_norm_clip, noise_multiplier):
    return DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,  # Use microbatches to account for privacy spending
        learning_rate=learning_rate
    )

def train_with_dpsgd(source_imgs, save_path, EPOCHS_set, BATCH_SIZE, noise_multiplier, l2_norm_clip, logging_period=1):
    start_time = datetime.now()
    EPOCHS = EPOCHS_set
    print('Epochs: %d, Batch size: %d' % (EPOCHS, BATCH_SIZE))
    
    num_imgs = source_imgs.shape[0]
    n_batches = num_imgs // BATCH_SIZE
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    # Create models
    G = Generator('Generator')
    D1 = Discriminator1('Discriminator1')
    D2 = Discriminator2('Discriminator2')

    current_iter = tf.Variable(0, trainable=False)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=int(n_batches), decay_rate=DECAY_RATE, staircase=False)

    # Create DP optimizers
    G_solver = create_dp_optimizer(learning_rate, l2_norm_clip, noise_multiplier)
    D1_solver = create_dp_optimizer(learning_rate, l2_norm_clip, noise_multiplier)
    D2_solver = create_dp_optimizer(learning_rate, l2_norm_clip, noise_multiplier)

    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(G=G, D1=D1, D2=D2, G_solver=G_solver, D1_solver=D1_solver, D2_solver=D2_solver, current_iter=current_iter)
    manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=500)

    @tf.function
    def train_G(VIS_batch, ir_batch):
        def G_loss_fn():
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                
            ir_batch_ds = tf.nn.avg_pool(ir_batch, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            diff = generated_img_ds - ir_batch_ds
            grad_of_vis = compute_gradient(VIS_batch)
                
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)
                
            G_loss_GAN_D1 = -tf.reduce_mean(tf.math.log(D1_fake + eps))
            G_loss_GAN_D2 = -tf.reduce_mean(tf.math.log(D2_fake + eps))
            G_loss_GAN = 1.3 * G_loss_GAN_D1 + 0.7 * G_loss_GAN_D2

            LOSS_IR = Fro_LOSS(diff)
            LOSS_VIS = L1_LOSS(compute_gradient(generated_img) - grad_of_vis)

            G_loss_norm = 1.0 * LOSS_IR + 1.0 * LOSS_VIS
            G_loss = G_loss_GAN + 0.8 * G_loss_norm
            return G_loss

        # Minimize the generator loss using the DP optimizer
        G_solver.minimize(G_loss_fn, var_list=G.trainable_variables)

    @tf.function
    def train_D1(VIS_batch, ir_batch):
        def D1_loss_fn():
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            D1_real = D1(VIS_batch, training=True)
            D1_fake = D1(generated_img, training=True)
                
            D1_loss_real = -tf.reduce_mean(tf.math.log(D1_real + eps))
            D1_loss_fake = -tf.reduce_mean(tf.math.log(1. - D1_fake + eps))
            D1_loss = D1_loss_fake + D1_loss_real
            return D1_loss

        # Minimize the D1 discriminator loss using the DP optimizer
        D1_solver.minimize(D1_loss_fn, var_list=D1.trainable_variables)

    @tf.function
    def train_D2(VIS_batch, ir_batch):
        def D2_loss_fn():
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            generated_img_ds = g0
                    
            D2_real = D2(ir_batch, training=True)
            D2_fake = D2(generated_img_ds, training=True)
                
            D2_loss_real = -tf.reduce_mean(tf.math.log(D2_real + eps))
            D2_loss_fake = -tf.reduce_mean(tf.math.log(1. - D2_fake + eps))
            D2_loss = D2_loss_fake + D2_loss_real
            return D2_loss

        # Minimize the D2 discriminator loss using the DP optimizer
        D2_solver.minimize(D2_loss_fn, var_list=D2.trainable_variables)

    # Start training
    for epoch in range(EPOCHS):
        np.random.shuffle(source_imgs)
        for batch in range(n_batches):
            current_iter.assign_add(1)

            # Get batch of visible and infrared images
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            ir_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]

            # Resize both visible and infrared patches to 84x84
            VIS_batch = np.array([ndimage.zoom(vis_patch, (patch_size / vis_patch.shape[0], patch_size / vis_patch.shape[1])) for vis_patch in VIS_batch])
            ir_batch = np.array([ndimage.zoom(ir_patch, (patch_size / ir_patch.shape[0], patch_size / ir_patch.shape[1])) for ir_patch in ir_batch])

            VIS_batch = np.expand_dims(VIS_batch, -1)
            ir_batch = np.expand_dims(ir_batch, -1)

            VIS_batch = tf.convert_to_tensor(VIS_batch, dtype=tf.float32)
            ir_batch = tf.convert_to_tensor(ir_batch, dtype=tf.float32)

            # Alternate between training the generators and discriminators
            if batch % 2 == 0:
                train_D1(VIS_batch, ir_batch)
                train_D2(VIS_batch, ir_batch)
            else:
                train_G(VIS_batch, ir_batch)

            if batch % logging_period == 0:
                elapsed_time = datetime.now() - start_time
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch}/{n_batches}, Elapsed Time: {elapsed_time}")

    # Save final checkpoint
    manager.save()
    print("Training completed.")
