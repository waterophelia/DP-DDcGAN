# This code implements a custom DeconvLayer (deconvolution layer) 
# It applies transposed convolution to increase spatial resolution

import tensorflow as tf

# Standard deviation for initializing weights in the deconvolution layer
WEIGHT_INIT_STDDEV = 0.1

class DeconvLayer(tf.keras.layers.Layer):
     # Initialize the deconvolutional layer with input and output filters, kernel size, strides, and a name scope
    def __init__(self, input_filters, output_filters, kernel_size, strides, scope_name):
        super(DeconvLayer, self).__init__()
        # Save strides and scope name
        self.strides = strides
        self.scope_name = scope_name
        # Initialize kernel (filters) using the helper function _create_variables
        self.kernel = self._create_variables(input_filters, output_filters, kernel_size, self.scope_name)
        
    # Helper function to create kernel (filter) variables
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        # Define the shape of the kernel as [kernel_height, kernel_width, output_channels, input_channels]
        shape = [kernel_size, kernel_size, output_filters, input_filters]
        # Add a weight variable with TruncatedNormal initialization using the defined standard deviation
        kernel = self.add_weight(name='kernel', shape=shape,
                                 initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV))
        return kernel
        
    # Forward pass through the layer
    def call(self, inputs):
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        # Calculate the output shape based on input shape and strides
        output_shape = [batch_size, input_shape[1] * self.strides[1], input_shape[2] * self.strides[2], self.kernel.shape[2]]
        
        # Perform the transpose convolution (deconvolution) operation
        return tf.nn.conv2d_transpose(inputs, filters=self.kernel,
                                      output_shape=output_shape, strides=self.strides, padding='SAME')
        
# Function for deconvolution specific to infrared (IR) input
def deconv_ir(input, strides, scope_name):
    # Create a DeconvLayer for IR input with 1 input and output filter, and kernel size of 3
    deconv_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=strides, scope_name=scope_name)
    # Apply the deconvolution layer to the input
    return deconv_layer(input)
# Function for deconvolution specific to visible (Vis) input
def deconv_vis(input, strides, scope_name):
    # Create a DeconvLayer for Vis input, similarly to the IR deconvolution layer
    deconv_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=strides, scope_name=scope_name)
    # Apply the deconvolution layer to the input
    return deconv_layer(input)
