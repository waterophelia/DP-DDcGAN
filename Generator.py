# This code implements a generator network for image translation using visible and infrared images
# It includes an encoder-decoder architecture and custom deconvolution layers

import tensorflow as tf
from Deconv import DeconvLayer

# Initialize weight standard deviation for truncated normal initialization
WEIGHT_INIT_STDDEV = 0.05

# Generator class that combines the encoder, decoder, and deconvolutional layers for visible and infrared images
class Generator(tf.keras.Model):
    def __init__(self, sco):
        super(Generator, self).__init__()
        # Initialize encoder and decoder
        self.encoder = Encoder(sco)
        self.decoder = Decoder(sco)
        # Deconvolution layers for infrared and visible images with specified kernel sizes and strides
        self.deconv_ir_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=12, strides=[1, 4, 4, 1], scope_name='deconv_ir')
        self.deconv_vis_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=[1, 1, 1, 1], scope_name='deconv_vis')
   
    # Forward pass of the generator
    def call(self, vis, ir):
        # Apply deconvolution for infrared and visible inputs
        IR = self.deconv_ir_layer(ir)
        VIS = self.deconv_vis_layer(vis)
        # Ensure the infrared image matches the size of the visible image by resizing if necessary
        if IR.shape[1] != VIS.shape[1] or IR.shape[2] != VIS.shape[2]:
            IR = tf.image.resize(IR, [VIS.shape[1], VIS.shape[2]])
        # Concatenate the visible and infrared images along the channel dimension
        img = tf.concat([VIS, IR], 3)
        # Encode the concatenated image to a latent code
        code = self.encoder(img)
        # Decode the latent code to generate the final image
        generated_img = self.decoder(code)
        return generated_img

# Encoder class that processes input images into a latent code
class Encoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Encoder, self).__init__(name=scope_name)
        self.scope = scope_name
        self.weight_vars = []  # Store convolutional weights
        self.bn_layers = []  # Store batch normalization layers
        with tf.name_scope(self.scope):
            with tf.name_scope('encoder'):
                # Create convolutional layers for the encoder
                self.weight_vars.append(self._create_variables(2, 48, 3, 'conv1_1'))
                self.weight_vars.append(self._create_variables(48, 48, 3, 'dense_block_conv1'))
                self.weight_vars.append(self._create_variables(96, 48, 3, 'dense_block_conv2'))
                self.weight_vars.append(self._create_variables(144, 48, 3, 'dense_block_conv3'))
                self.weight_vars.append(self._create_variables(192, 48, 3, 'dense_block_conv4'))
                # Initialize batch normalization layers for each convolutional layer except the first
                for i in range(len(self.weight_vars)):
                    self.bn_layers.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))

    # Helper function to create convolutional weights and biases
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.name_scope(scope):
            kernel = self.add_weight(shape=shape, initializer=tf.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = self.add_weight(shape=[output_filters], initializer='zeros', name='bias')
        return (kernel, bias)

    # Forward pass through the encoder
    def call(self, image, training=True):
        dense_indices = [1, 2, 3, 4, 5]  # Dense connections after each dense block
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            # Apply convolution, batch normalization, and ReLU activation
            out = conv2d(out, kernel, bias, self.bn_layers[i], dense=(i in dense_indices), use_relu=True, Scope=f'{self.scope}/encoder/b{i}', training=training)
        return out

# Decoder class that processes latent code into the final image
class Decoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Decoder, self).__init__(name=scope_name)
        self.weight_vars = []  # Store convolutional weights
        self.bn_layers = []  # Store batch normalization layers
        self.scope = scope_name
        with tf.name_scope(scope_name):
            with tf.name_scope('decoder'):
                # Create convolutional layers for the decoder
                self.weight_vars.append(self._create_variables(240, 240, 3, 'conv2_1'))
                self.weight_vars.append(self._create_variables(240, 128, 3, 'conv2_2'))
                self.weight_vars.append(self._create_variables(128, 64, 3, 'conv2_3'))
                self.weight_vars.append(self._create_variables(64, 32, 3, 'conv2_4'))
                self.weight_vars.append(self._create_variables(32, 1, 3, 'conv2_5'))
                # Initialize batch normalization layers for all but the last layer
                for i in range(len(self.weight_vars) - 1):
                    self.bn_layers.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))

    # Helper function to create convolutional weights and biases
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.name_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = self.add_weight(shape=shape, initializer=tf.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = self.add_weight(shape=[output_filters], initializer='zeros', name='bias')
        return (kernel, bias)

    # Forward pass through the decoder
    def call(self, image, training=True):
        final_layer_idx = len(self.weight_vars) - 1  # Index of the last layer
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            # Apply convolution, batch normalization (except for the first and last layers), and activation
            if i == 0:
                out = conv2d(out, kernel, bias, None, dense=False, use_relu=True, Scope=f'{self.scope}/decoder/b{i}', BN=False, training=training)
            elif i == final_layer_idx:
                # Final layer uses tanh activation to generate the image
                out = conv2d(out, kernel, bias, None, dense=False, use_relu=False, Scope=f'{self.scope}/decoder/b{i}', BN=False, training=training)
                out = tf.nn.tanh(out) / 2 + 0.5  # Rescale tanh output to [0, 1]
            else:
                out = conv2d(out, kernel, bias, self.bn_layers[i-1], dense=False, use_relu=True, BN=True, Scope=f'{self.scope}/decoder/b{i}', training=training)
        return out

import tensorflow as tf
from Deconv import DeconvLayer

# Initialize weight standard deviation for truncated normal initialization
WEIGHT_INIT_STDDEV = 0.05

# Generator class that combines the encoder, decoder, and deconvolutional layers for visible and infrared images
class Generator(tf.keras.Model):
    def __init__(self, sco):
        super(Generator, self).__init__()
        # Initialize encoder and decoder
        self.encoder = Encoder(sco)
        self.decoder = Decoder(sco)
        # Deconvolution layers for infrared and visible images with specified kernel sizes and strides
        self.deconv_ir_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=12, strides=[1, 4, 4, 1], scope_name='deconv_ir')
        self.deconv_vis_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=[1, 1, 1, 1], scope_name='deconv_vis')
   
    # Forward pass of the generator
    def call(self, vis, ir):
        # Apply deconvolution for infrared and visible inputs
        IR = self.deconv_ir_layer(ir)
        VIS = self.deconv_vis_layer(vis)
        # Ensure the infrared image matches the size of the visible image by resizing if necessary
        if IR.shape[1] != VIS.shape[1] or IR.shape[2] != VIS.shape[2]:
            IR = tf.image.resize(IR, [VIS.shape[1], VIS.shape[2]])
        # Concatenate the visible and infrared images along the channel dimension
        img = tf.concat([VIS, IR], 3)
        # Encode the concatenated image to a latent code
        code = self.encoder(img)
        # Decode the latent code to generate the final image
        generated_img = self.decoder(code)
        return generated_img

# Encoder class that processes input images into a latent code
class Encoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Encoder, self).__init__(name=scope_name)
        self.scope = scope_name
        self.weight_vars = []  # Store convolutional weights
        self.bn_layers = []  # Store batch normalization layers
        with tf.name_scope(self.scope):
            with tf.name_scope('encoder'):
                # Create convolutional layers for the encoder
                self.weight_vars.append(self._create_variables(2, 48, 3, 'conv1_1'))
                self.weight_vars.append(self._create_variables(48, 48, 3, 'dense_block_conv1'))
                self.weight_vars.append(self._create_variables(96, 48, 3, 'dense_block_conv2'))
                self.weight_vars.append(self._create_variables(144, 48, 3, 'dense_block_conv3'))
                self.weight_vars.append(self._create_variables(192, 48, 3, 'dense_block_conv4'))
                # Initialize batch normalization layers for each convolutional layer except the first
                for i in range(len(self.weight_vars)):
                    self.bn_layers.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))

    # Helper function to create convolutional weights and biases
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.name_scope(scope):
            kernel = self.add_weight(shape=shape, initializer=tf.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = self.add_weight(shape=[output_filters], initializer='zeros', name='bias')
        return (kernel, bias)

    # Forward pass through the encoder
    def call(self, image, training=True):
        dense_indices = [1, 2, 3, 4, 5]  # Dense connections after each dense block
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            # Apply convolution, batch normalization, and ReLU activation
            out = conv2d(out, kernel, bias, self.bn_layers[i], dense=(i in dense_indices), use_relu=True, Scope=f'{self.scope}/encoder/b{i}', training=training)
        return out

# Decoder class that processes latent code into the final image
class Decoder(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Decoder, self).__init__(name=scope_name)
        self.weight_vars = []  # Store convolutional weights
        self.bn_layers = []  # Store batch normalization layers
        self.scope = scope_name
        with tf.name_scope(scope_name):
            with tf.name_scope('decoder'):
                # Create convolutional layers for the decoder
                self.weight_vars.append(self._create_variables(240, 240, 3, 'conv2_1'))
                self.weight_vars.append(self._create_variables(240, 128, 3, 'conv2_2'))
                self.weight_vars.append(self._create_variables(128, 64, 3, 'conv2_3'))
                self.weight_vars.append(self._create_variables(64, 32, 3, 'conv2_4'))
                self.weight_vars.append(self._create_variables(32, 1, 3, 'conv2_5'))
                # Initialize batch normalization layers for all but the last layer
                for i in range(len(self.weight_vars) - 1):
                    self.bn_layers.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))

    # Helper function to create convolutional weights and biases
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.name_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = self.add_weight(shape=shape, initializer=tf.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = self.add_weight(shape=[output_filters], initializer='zeros', name='bias')
        return (kernel, bias)

    # Forward pass through the decoder
    def call(self, image, training=True):
        final_layer_idx = len(self.weight_vars) - 1  # Index of the last layer
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            # Apply convolution, batch normalization (except for the first and last layers), and activation
            if i == 0:
                out = conv2d(out, kernel, bias, None, dense=False, use_relu=True, Scope=f'{self.scope}/decoder/b{i}', BN=False, training=training)
            elif i == final_layer_idx:
                # Final layer uses tanh activation to generate the image
                out = conv2d(out, kernel, bias, None, dense=False, use_relu=False, Scope=f'{self.scope}/decoder/b{i}', BN=False, training=training)
                out = tf.nn.tanh(out) / 2 + 0.5  # Rescale tanh output to [0, 1]
            else:
                out = conv2d(out, kernel, bias, self.bn_layers[i-1], dense=False, use_relu=True, BN=True, Scope=f'{self.scope}/decoder/b{i}', training=training)
        return out

# Helper function for applying convolution, bias, batch normalization, ReLU, and optional dense connections
def conv2d(x, kernel, bias, bn_layer=None, dense=False, use_relu=True, Scope=None, BN=True, training=True):
    # Pad the input using reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    # Apply convolution
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    # Optionally apply batch normalization
    if BN and bn_layer is not None:
        with tf.name_scope(Scope):
            out = bn_layer(out, training=training)
    # Optionally apply ReLU activation
    if use_relu:
        out = tf.nn.relu(out)
    # If using dense (skip) connections, concatenate the output with the input
    if dense:
        out = tf.concat([out, x], 3)
    return out

