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

tf.config.run_functions_eagerly(True)

patch_size = 84
LEARNING_RATE = 0.0002
DECAY_RATE = 0.9
eps = 1e-8
rc = 4

def compute_gradient(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g

def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period=1, image_save_period=100, image_save_path='./generated_images'):
    start_time = datetime.now()
    EPOCHS = EPOCHES_set
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
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=int(n_batches), decay_rate=DECAY_RATE, staircase=False)

    G_GAN_solver = tf.keras.optimizers.RMSprop(learning_rate)
    G_solver = tf.keras.optimizers.RMSprop(learning_rate)
    D1_solver = tf.keras.optimizers.SGD(learning_rate)
    D2_solver = tf.keras.optimizers.SGD(learning_rate)

    # Set up checkpoints
    checkpoint = tf.train.Checkpoint(G=G, D1=D1, D2=D2, G_GAN_solver=G_GAN_solver, G_solver=G_solver, 
                                     D1_solver=D1_solver, D2_solver=D2_solver, current_iter=current_iter)
    manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=500)

    def save_generated_image(image, epoch, batch, save_path):
      os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
      image = tf.squeeze(image, axis=-1)  # Remove the last channel if it's 1
      plt.figure(figsize=(4, 4))
      plt.imshow(image, cmap='gray')
      plt.axis('off')
      plt.savefig(os.path.join(save_path, f'epoch_{epoch}_batch_{batch}.png'))
      plt.close()

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
            
            G_loss_GAN_D1 = -tf.reduce_mean(tf.math.log(D1_fake + eps))
            G_loss_GAN_D2 = -tf.reduce_mean(tf.math.log(D2_fake + eps))
            G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2

            LOSS_IR = Fro_LOSS(diff)
            LOSS_VIS = L1_LOSS(compute_gradient(generated_img) - grad_of_vis)
            G_loss_norm = LOSS_IR + 1.2 * LOSS_VIS
            G_loss = G_loss_GAN + 0.8 * G_loss_norm

        gradients = tape.gradient(G_loss, G.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        G_solver.apply_gradients(zip(clipped_gradients, G.trainable_variables))
        
        return G_loss, G_loss_GAN_D1, G_loss_GAN_D2, D1_fake, D2_fake, generated_img

    @tf.function
    def train_D1(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)

            # Pass through the discriminator
            D1_real = D1(VIS_batch, training=True)
            D1_fake = D1(generated_img, training=True)
            
            D1_loss_real = -tf.reduce_mean(tf.math.log(D1_real + eps))
            D1_loss_fake = -tf.reduce_mean(tf.math.log(1. - D1_fake + eps))
            D1_loss = D1_loss_fake + D1_loss_real

        gradients = tape.gradient(D1_loss, D1.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        D1_solver.apply_gradients(zip(clipped_gradients, D1.trainable_variables))
        
        return D1_loss, D1_real, D1_fake

    @tf.function
    def train_D2(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            generated_img_ds = g0  # Output shape should be [batch_size, 21, 21, channels]
                
            D2_real = D2(ir_batch, training=True)
            D2_fake = D2(generated_img_ds, training=True)
            
            D2_loss_real = -tf.reduce_mean(tf.math.log(D2_real + eps))
            D2_loss_fake = -tf.reduce_mean(tf.math.log(1. - D2_fake + eps))
            D2_loss = D2_loss_fake + D2_loss_real

        gradients = tape.gradient(D2_loss, D2.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        D2_solver.apply_gradients(zip(clipped_gradients, D2.trainable_variables))
        
        return D2_loss, D2_real, D2_fake

    @tf.function
    def train_G_GAN(VIS_batch, ir_batch):
        with tf.GradientTape() as tape:
            generated_img = G(vis=VIS_batch, ir=ir_batch)
            g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            D1_fake = D1(generated_img, training=True)
            D2_fake = D2(generated_img_ds, training=True)

            G_loss_GAN_D1 = -tf.reduce_mean(tf.math.log(D1_fake + eps))
            G_loss_GAN_D2 = -tf.reduce_mean(tf.math.log(D2_fake + eps))
            G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2

        gradients = tape.gradient(G_loss_GAN, G.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -8, 8) for grad in gradients]
        G_GAN_solver.apply_gradients(zip(clipped_gradients, G.trainable_variables))
        
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
                g_loss_gan_d1 = 0  # Placeholder value
                g_loss_gan_d2 = 0  # Placeholder value
          else:
              # Call train_G and get the generated image
              g_loss, g_loss_gan_d1, g_loss_gan_d2, d1_fake, d2_fake, generated_img = train_G(VIS_batch, ir_batch)
              it_g += 1
              d1_loss = d2_loss = 0  # Placeholder values

              # Save the generated image periodically
              if batch % image_save_period == 0:
                  save_generated_image(generated_img[0], epoch, batch, save_path)  # Save the first image in the batch

          if batch % 2 == 0:
              while d1_loss > 1.9 and it_d1 < 20:
                  d1_loss, _, _ = train_D1(VIS_batch, ir_batch)
                  it_d1 += 1
              while d2_loss > 1.9 and it_d2 < 20:
                  d2_loss, _, _ = train_D2(VIS_batch, ir_batch)
                  it_d2 += 1
          else:
              while (d1_loss < 1 or d2_loss < 1) and it_g < 20:
                  g_loss_gan = train_G_GAN(VIS_batch, ir_batch)
                  g_loss, g_loss_gan_d1, g_loss_gan_d2, d1_fake, d2_fake, generated_img = train_G(VIS_batch, ir_batch)
                  it_g += 1
              while g_loss > 200 and it_g < 20:
                  g_loss, _, _, _, _, _ = train_G(VIS_batch, ir_batch)
                  it_g += 1

          if batch % 1 == 0:
                elapsed_time = datetime.now() - start_time
                lr = learning_rate(current_iter)
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch}/{n_batches}, G_loss: {g_loss}, D1_loss: {d1_loss}, D2_loss: {d2_loss}")
                print(f"Learning Rate: {lr}, Elapsed Time: {elapsed_time}\n")

          if step % logging_period == 0:
                manager.save()
            
          is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
          if is_last_step or step % logging_period == 0:
                elapsed_time = datetime.now() - start_time
                lr = learning_rate(current_iter)
                print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
                    epoch + 1, EPOCHS, step, lr, elapsed_time))

    manager.save()
