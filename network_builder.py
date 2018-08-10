import tensorflow as tf


class NetworkBuilder:
    def add_conv_layer(self, input_layer, output_size=32, feature_size=(5, 5), strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu):

        with tf.name_scope("Convolution"):
            return tf.layers.conv2d(inputs=input_layer,
                                    filters=output_size,
                                    kernel_size=feature_size,
                                    padding=padding, activation=activation)

    def add_max_pooling_layer(self, input_layer, pool_size=(2, 2), strides=2):

        with tf.name_scope("Pooling"):
            return tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size, strides=strides)

    def flatten(self, input_layer):

        with tf.name_scope("Flatten"):
            input_size = input_layer.get_shape().as_list()
            new_size = input_size[-1] * input_size[-2] * input_size[-3]
            return tf.reshape(input_layer, [-1, new_size])

    def add_dense_layer(self, input_layer, units, activation=None):

        with tf.name_scope("DenseLayer"):
            if activation == None:
                return tf.layers.dense(inputs=input_layer, units=units)
            else:
                return tf.layers.dense(inputs=input_layer, units=units, activation=activation)

    def add_dropout(self, input_layer, rate, training):

        with tf.name_scope("Dropout"):
            return tf.layers.dropout(inputs=input_layer, rate=rate, training=training)

    def add_batch_normalization(self, input_layer, training):

        with tf.name_scope("BatchNormalization"):
            return tf.layers.batch_normalization(input_layer, training=training)
