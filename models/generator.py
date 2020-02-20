import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, num_input, num_output, layer_sizes):
        super(Generator, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.layer_sizes = layer_sizes

        self.layers_list = []

        self.layers_list.append(tf.keras.layers.Dense(self.layer_sizes[0], input_shape=(self.num_input,)))
        bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.layers_list.append(bn)

        activation = tf.keras.layers.ReLU()
        self.layers_list.append(activation)

        for i in range(len(self.layer_sizes)):
            layer = tf.keras.layers.Dense(self.layer_sizes[i])
            self.layers_list.append(layer)

            bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.layers_list.append(bn)

            activation = tf.keras.layers.ReLU()
            self.layers_list.append(activation)

        self.layers_list.append(tf.keras.layers.Dense(self.num_output))

    def call(self, x, training):
        for i, layer in enumerate(self.layers_list):
            if i in [1, 4, 7] and not training:
                continue
            x = layer(x)

        return x

    def anomaly_generate(self, x, training):
        for i, layer in enumerate(self.layers_list):
            if i in [1, 4, 7] and not training:
                continue
            layer.trainable
            x = layer(x)

        return x
