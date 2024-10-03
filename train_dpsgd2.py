import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from scipy import ndimage
from datetime import datetime
import os

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from generate import generate

#tf.config.run_functions_eagerly(True)

patch_size = 84
LEARNING_RATE = 0.0002
eps = 1e-8
rc = 4

def compute_gradient(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g

def train_with_dpsgd(source_imgs, save_path, EPOCHS_set, BATCH_SIZE,  noise_multiplier, l2_norm_clip, logging_period=1):
    start_time = datetime.now()
    EPOCHS = EPOCHS_set
    print('Epochs: %d, Batch size: %d' % (EPOCHS, BATCH_SIZE))
    
    checkpoint_save_path = save_path + 'temporary.ckpt'
    num_imgs = source_imgs.shape[0]
    mod = num_imgs % BATCH_SIZE
    n_batches = int(num_imgs // BATCH_SIZE)
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source_imgs = source_imgs[:-mod]

    # Create the models
    G = Generator('Generator')
    D1 = Discriminator1('Discriminator1')
    D2 = Discriminator2('Discriminator2')

    current_iter = tf.Variable(0, trainable=False)

    # Use DP-SGD optimizer
    dp_optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,  # Maximum L2 norm for gradients
        noise_multiplier=noise_multiplier,  # Noise multiplier for DP
        num_microbatches=BATCH_SIZE,  # Split the batch into microbatches
        learning_rate=LEARNING_RATE  # Learning rate for the optimizer
    )

     # Set up checkpoints
    checkpoint = tf.train.Checkpoint(G=G, D1=D1, D2=D2, G_solver=dp_optimizer, D1_solver=dp_optimizer, D2_solver=dp_optimizer, current_iter=current_iter)
    manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=500)

    @tf.function
    def train_G(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # Downsample ir_batch to match the dimensions of generated_img_ds
            ir_batch_ds = tf.nn.avg_pool(ir_batch, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            # Compute difference and inspect
            diff = generated_img_ds - ir_batch_ds
            grad_of_vis = compute_gradient(VIS_batch)  # Using the grad function defined above
            
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)
            
            G_loss_GAN_D1 = -tf.math.log(D1_fake + eps)
            G_loss_GAN_D2 = -tf.math.log(D2_fake + eps)
            G_loss_GAN = 1.3 * G_loss_GAN_D1 + 0.7 * G_loss_GAN_D2

            LOSS_IR = Fro_LOSS(diff)
            LOSS_VIS = L1_LOSS(compute_gradient(generated_img) - grad_of_vis)

            G_loss_norm = 1.0 * LOSS_IR + 1.0 * LOSS_VIS
            G_loss = G_loss_GAN + 0.8 * G_loss_norm

        # Use the DP-SGD optimizer's compute gradients function to ensure differential privacy
        gradients = dp_optimizer._compute_gradients(G_loss, G.trainable_variables, tape=tape)
        dp_optimizer.apply_gradients(zip(gradients, G.trainable_variables))

        return G_loss, G_loss_GAN_D1, G_loss_GAN_D2, D1_fake, D2_fake, generated_img

    @tf.function
    def train_D1(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)

            print(VIS_batch.shape)
            print(generated_img.shape)
            print(D1.trainable_variables[0].shape)

            # Pass through the discriminator
            D1_real = D1(VIS_batch, training=True)
            D1_fake = D1(generated_img, training=True)
            
            D1_loss_real = -tf.math.log(D1_real + eps)
            D1_loss_fake = -tf.math.log(1. - D1_fake + eps)
            D1_loss = D1_loss_fake + D1_loss_real

        gradients = dp_optimizer._compute_gradients(D1_loss, D1.trainable_variables, tape=tape)
        dp_optimizer.apply_gradients(zip(gradients, D1.trainable_variables))
        
        return D1_loss, D1_real, D1_fake

    @tf.function
    def train_D2(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            generated_img_ds = g0  # Output shape should be [batch_size, 21, 21, channels]
                
            D2_real = D2(ir_batch, training=True)
            D2_fake = D2(generated_img_ds, training=True)
            
            D2_loss_real = -tf.math.log(D2_real + eps)
            D2_loss_fake = -tf.math.log(1. - D2_fake + eps)
            D2_loss = D2_loss_fake + D2_loss_real

        gradients = dp_optimizer._compute_gradients(D2_loss, D2.trainable_variables, tape=tape)
        dp_optimizer.apply_gradients(zip(gradients, D2.trainable_variables))
        
        return D2_loss, D2_real, D2_fake

    @tf.function
    def train_G_GAN(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)

            G_loss_GAN_D1 = -tf.math.log(D1_fake + eps)
            G_loss_GAN_D2 = -tf.math.log(D2_fake + eps)
            G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2

        gradients = dp_optimizer._compute_gradients(G_loss_GAN, G.trainable_variables, tape=tape)
        dp_optimizer.apply_gradients(zip(gradients, G.trainable_variables))
        
        return G_loss_GAN

    # ** Start Training **
    step = 0
    count_loss = 0
    num_imgs = source_imgs.shape[0]

    for epoch in range(EPOCHS):
        np.random.shuffle(source_imgs)
        for batch in range(n_batches):
          step += 1
          current_iter.assign(step)

          # Extract and resize the batch
          VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
          ir_or_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
          
          # Resizing both the visible and infrared patches to 84x84
          VIS_batch = np.array([ndimage.zoom(vis_patch, (patch_size / vis_patch.shape[0], patch_size / vis_patch.shape[1])) for vis_patch in VIS_batch])
          ir_batch = np.array([ndimage.zoom(ir_patch, (patch_size / ir_patch.shape[0], patch_size / ir_patch.shape[1])) for ir_patch in ir_or_batch])

          # Expand dimensions to match the expected input format for the model
          VIS_batch = np.expand_dims(VIS_batch, -1)
          ir_batch = np.expand_dims(ir_batch, -1)

          # Convert to TensorFlow tensors
          VIS_batch = tf.convert_to_tensor(VIS_batch, dtype=tf.float32)
          ir_batch = tf.convert_to_tensor(ir_batch, dtype=tf.float32)

          it_g = 0
          it_d1 = 0
          it_d2 = 0

          if batch % 2 == 0:
                d1_loss, d1_real, d1_fake = train_D1(VIS_batch, ir_batch)
                it_d1 += 1
                d2_loss, d2_real, d2_fake = train_D2(VIS_batch, ir_batch)
                it_d2 += 1
                g_loss = 0  # Placeholder value
                g_loss_GAN_D1 = 0  # Placeholder value
                g_loss_GAN_D2 = 0  # Placeholder value
          else:
              # Call train_G and get the generated image
              g_loss, g_loss_GAN_d1, g_loss_GAN_d2, D1_fake, D2_fake, generated_img = train_G(VIS_batch, ir_batch)
              it_g += 1
              d1_loss = d2_loss = 0  # Placeholder values

          if batch % 2 == 0:
              while d1_loss > 1.9 and it_d1 < 20:
                  d1_loss, _, _ = train_D1(VIS_batch, ir_batch)
                  it_d1 += 1
              while d2_loss > 1.9 and it_d2 < 20:
                  d2_loss, _, _ = train_D2(VIS_batch, ir_batch)
                  it_d2 += 1
          else:
              while (d1_loss < 1 or d2_loss < 1) and it_g < 20:
                  g_loss_GAN = train_G_GAN(VIS_batch, ir_batch)
                  g_loss, g_loss_GAN_d1, g_loss_GAN_d2, d1_fake, d2_fake, generated_img = train_G(VIS_batch, ir_batch)
                  it_g += 1
              while g_loss > 200 and it_g < 20:
                  g_loss, _, _, _, _, _ = train_G(VIS_batch, ir_batch)
                  it_g += 1

          if batch % 10 == 0:
                epsilon, delta = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
                    n=len(source_imgs),
                    batch_size=BATCH_SIZE,
                    noise_multiplier=noise_multiplier,
                    epochs=(epoch + 1) + (batch / n_batches),
                    delta=1e-5
                )
                elapsed_time = datetime.now() - start_time
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch}/{n_batches}, Epsilon = {epsilon}, delta = 1e-5, G_loss: {g_loss}, D1_loss: {d1_loss}, D2_loss: {d2_loss}")
                print(f"Elapsed Time: {elapsed_time}\n")

          if step % logging_period == 0:
                manager.save()

    manager.save()
print("Training completed.")
