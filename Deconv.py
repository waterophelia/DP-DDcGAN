import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

class DeconvLayer(tf.keras.layers.Layer):
    def __init__(self, input_filters, output_filters, kernel_size, strides, scope_name):
        super(DeconvLayer, self).__init__()
        self.strides = strides
        self.scope_name = scope_name
        self.kernel = self._create_variables(input_filters, output_filters, kernel_size, self.scope_name)

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, output_filters, input_filters]
        kernel = self.add_weight(name='kernel', shape=shape,
                                 initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV))
        return kernel

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        output_shape = [batch_size, input_shape[1] * self.strides[1], input_shape[2] * self.strides[2], self.kernel.shape[2]]
        
        return tf.nn.conv2d_transpose(inputs, filters=self.kernel,
                                      output_shape=output_shape, strides=self.strides, padding='SAME')

def deconv_ir(input, strides, scope_name):
    deconv_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=strides, scope_name=scope_name)
    return deconv_layer(input)

def deconv_vis(input, strides, scope_name):
    deconv_layer = DeconvLayer(input_filters=1, output_filters=1, kernel_size=3, strides=strides, scope_name=scope_name)
    return deconv_layer(input)
