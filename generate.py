import numpy as np
from imageio import imwrite, imread  # Updated for correct saving
from os.path import join
import tensorflow as tf
from Generator import Generator

def generate(ir_path, vis_path, model_path, index, output_path=None):
    # Load and preprocess images
    ir_img = imread(ir_path) / 255.0
    vis_img = imread(vis_path) / 255.0

    ir_dimension = list(ir_img.shape)
    vis_dimension = list(vis_img.shape)
    ir_dimension.insert(0, 1)  # Add batch dimension
    ir_dimension.append(1)  # Add channel dimension
    vis_dimension.insert(0, 1)  # Add batch dimension
    vis_dimension.append(1)  # Add channel dimension

    ir_img = ir_img.reshape(ir_dimension)
    vis_img = vis_img.reshape(vis_dimension)

    # Use tf.function to create a graph-like execution
    @tf.function
    def run_model(SOURCE_VIS, SOURCE_ir):
        G = Generator('Generator')
        output_image = G(vis=SOURCE_VIS, ir=SOURCE_ir)
        return output_image

    # Create variables for input
    SOURCE_VIS = tf.Variable(vis_img, dtype=tf.float32)
    SOURCE_ir = tf.Variable(ir_img, dtype=tf.float32)

    # Restore the trained model
    G = Generator('Generator')
    checkpoint = tf.train.Checkpoint(G=G)
    checkpoint.restore(tf.train.latest_checkpoint(model_path)).expect_partial()

    # Run the model
    output_image = run_model(SOURCE_VIS, SOURCE_ir)

    # Generate output
    output = output_image.numpy()
    output = output[0, :, :, 0]  # Remove batch and channel dimensions

    # Convert the output to uint8 format for saving as BMP
    output = (output * 255).astype(np.uint8)  # Scale to [0, 255] and cast to uint8

    # Save the output as BMP
    imwrite(join(output_path, f"{index}.bmp"), output)
