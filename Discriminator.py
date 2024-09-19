import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

class Discriminator1(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Discriminator1, self).__init__()
        self.scope = scope_name
        
        # Create weights in the constructor
        self.conv_layers = [
            self._create_variables(1, 16, 3, scope='conv1'),
            self._create_variables(16, 32, 3, scope='conv2'),
            self._create_variables(32, 64, 3, scope='conv3')
        ]
        
        # Create BatchNormalization layers in the constructor
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization()
        ]

        # Dense layer
        self.flatten_dense = tf.keras.layers.Dense(1, activation='tanh')

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        #print(f"Creating variables for {scope}: input_filters={input_filters}, output_filters={output_filters}")
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

    def call(self, img, training=True):
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        #print(f"D1 input shape: {img.shape}")
        
        out = img
        for i, layer_vars in enumerate(self.conv_layers):
            # Only apply batch normalization on layers other than the first
            use_bn = (i > 0)
            out = self._conv2d_layer(out, layer_vars['kernel'], layer_vars['bias'], training=training, use_bn=use_bn, bn_layer=self.batch_norm_layers[i-1] if use_bn else None)
            #print(f"D1 after conv{i+1} shape: {out.shape}")
        
        out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
        #print(f"D1 after flatten shape: {out.shape}")
        out = self.flatten_dense(out)
        #print(f"D1 after dense shape: {out.shape}")
        out = out / 2 + 0.5
        return out

    def _conv2d_layer(self, x, kernel, bias, training=True, use_bn=True, bn_layer=None):
        # Padding image with reflection mode
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 2, 2, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)
        if use_bn and bn_layer is not None:
            out = bn_layer(out, training=training)
        out = tf.nn.relu(out)
        return out

class Discriminator2(tf.keras.layers.Layer):
    def __init__(self, scope_name):
        super(Discriminator2, self).__init__()
        self.scope = scope_name
        self.conv_layers = []

        # Define convolutional layers
        with tf.name_scope(scope_name):
            self.conv_layers.append(self._create_conv_layer(1, 16, 3, 'conv1'))
            self.conv_layers.append(self._create_conv_layer(16, 32, 3, 'conv2'))
            self.conv_layers.append(self._create_conv_layer(32, 64, 3, 'conv3'))

        # Define BatchNormalization layers
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(2)]

        # Replace flattening with global average pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Define Dense layer
        self.flatten_dense = tf.keras.layers.Dense(1, activation='tanh', name=f'{self.scope}_flatten1')

    def _create_conv_layer(self, input_filters, output_filters, kernel_size, scope):
        return tf.keras.layers.Conv2D(output_filters, kernel_size, strides=(2, 2), padding='same',
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV),
                                      name=scope)

    def call(self, img, training=True):
        #print(f"D2 input shape: {img.shape}")
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        out = img
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            #print(f"D2 after conv{i+1} shape: {out.shape}")
            out = tf.nn.relu(out)
            if i > 0:
                out = self.batch_norm_layers[i-1](out, training=training)
                #print(f"D2 after BN{i} shape: {out.shape}")
        
        # Replace flattening with global average pooling
        out = self.global_avg_pool(out)
        #print(f"D2 after global average pooling shape: {out.shape}")
        
        out = self.flatten_dense(out)
        #print(f"D2 after dense shape: {out.shape}")
        out = out / 2 + 0.5
        return out

    def _conv2d_layer(self, x, kernel, bias, training=True, use_bn=True, bn_layer=None):
        # Padding image with reflection mode
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 2, 2, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)
        if use_bn and bn_layer is not None:
            out = bn_layer(out, training=training)
        out = tf.nn.relu(out)
        return out
