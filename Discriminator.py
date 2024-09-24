# This code implements two discriminators (Discriminator1 and Discriminator2) for DDcGAN
# These discriminators evaluate the generated images and apply convolutional layers with optional batch normalization

import tensorflow as tf

# Standard deviation for initializing weights in the convolutional layers
WEIGHT_INIT_STDDEV = 0.1

# Discriminator1: Implements a custom layer for the first discriminator
class Discriminator1(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Discriminator1, self).__init__()
        self.scope = scope_name
        
        # Define convolutional layers and create weights
        self.conv_layers = [
            self._create_variables(1, 16, 3, scope='conv1'), # Input: 1 filter, Output: 16 filters
            self._create_variables(16, 32, 3, scope='conv2'), # Input: 16 filters, Output: 32 filters
            self._create_variables(32, 64, 3, scope='conv3')  # Input: 32 filters, Output: 64 filters
        ]
        
         # Define BatchNormalization layers for the later convolutional layers
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization(), # For conv2
            tf.keras.layers.BatchNormalization() # For conv3
        ]

        # Dense layer that outputs a single value with tanh activation
        self.flatten_dense = tf.keras.layers.Dense(1, activation='tanh')

    # Helper function to create weights for the convolutional layers
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        return {
            'kernel': self.add_weight(
                name=f'{scope}_kernel',
                shape=[kernel_size, kernel_size, input_filters, output_filters],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)
            ),
            'bias': self.add_weight(
                name=f'{scope}_bias',
                shape=[output_filters],
                initializer='zeros'
            )
        }

    # Forward pass of the discriminator
    def call(self, img, training=True):
        # Ensure input has 4 dimensions (batch_size, height, width, channels)
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        
        out = img
        for i, layer_vars in enumerate(self.conv_layers):
            # Apply convolution and optional batch normalization
            use_bn = (i > 0) # Only apply batch normalization on layers other than the first
            out = self._conv2d_layer(out, layer_vars['kernel'], layer_vars['bias'], training=training, use_bn=use_bn, bn_layer=self.batch_norm_layers[i-1] if use_bn else None)
            #print(f"D1 after conv{i+1} shape: {out.shape}")

        # Flatten the output tensor
        out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
         # Apply the dense layer
        out = self.flatten_dense(out)
        # Adjust the output to be in the range [0, 1]
        out = out / 2 + 0.5
        return out

    # Helper function for convolutional layers with optional BatchNormalization
    def _conv2d_layer(self, x, kernel, bias, training=True, use_bn=True, bn_layer=None):
        # Padding the input using reflection mode
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        # Perform convolution
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 2, 2, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)
        # Apply batch normalization if required
        if use_bn and bn_layer is not None:
            out = bn_layer(out, training=training)
        # Apply ReLU activation
        out = tf.nn.relu(out)
        return out

# Discriminator2: Similar to Discriminator1 but uses Conv2D layers and global average pooling
class Discriminator2(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Discriminator2, self).__init__()
        self.scope = scope_name
        self.conv_layers = []

        # Define convolutional layers
        with tf.name_scope(scope_name):
            self.conv_layers.append(self._create_conv_layer(1, 16, 3, 'conv1')) # Input: 1 filter, Output: 16 filters
            self.conv_layers.append(self._create_conv_layer(16, 32, 3, 'conv2')) # Input: 16 filters, Output: 32 filters
            self.conv_layers.append(self._create_conv_layer(32, 64, 3, 'conv3')) # Input: 32 filters, Output: 64 filters

        # Define BatchNormalization layers
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(2)]

        # Replace flattening with global average pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Define Dense layer that outputs a single value with tanh activation
        self.flatten_dense = tf.keras.layers.Dense(1, activation='tanh', name=f'{self.scope}_flatten1')

     # Helper function to create Conv2D layers
    def _create_conv_layer(self, input_filters, output_filters, kernel_size, scope):
        return tf.keras.layers.Conv2D(output_filters, kernel_size, strides=(2, 2), padding='same',
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV),
                                      name=scope)

    # Forward pass of the discriminator
    def call(self, img, training=True):
        #print(f"D2 input shape: {img.shape}")
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        out = img
        # Apply convolutional layers with ReLU and batch normalization
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            #print(f"D2 after conv{i+1} shape: {out.shape}")
            out = tf.nn.relu(out)
            if i > 0: # Batch normalization for conv2 and conv3 layers
                out = self.batch_norm_layers[i-1](out, training=training)
        
        # Replace flattening with global average pooling
        out = self.global_avg_pool(out)

        # Apply the dense layer
        out = self.flatten_dense(out)
        
        # Adjust the output to be in the range [0, 1]
        out = out / 2 + 0.5
        return out

    def _conv2d_layer(self, x, kernel, bias, training=True, use_bn=True, bn_layer=None):
        ## Padding the input using reflection mode to maintain spatial dimensions 
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        # Apply the convolution operation
        # Strides: [1, 2, 2, 1] means the convolution window moves by 2 pixels in both height and width
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 2, 2, 1], padding='VALID')

        # Add the bias to the convolution output
        out = tf.nn.bias_add(out, bias)

        # If use_bn is True and a BatchNormalization layer is provided, apply batch normalization
        if use_bn and bn_layer is not None:
            out = bn_layer(out, training=training)

        # Apply ReLU activation to introduce non-linearity
        out = tf.nn.relu(out)
        return out
