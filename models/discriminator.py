import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, num_input, num_output, layer_sizes):
        super(Discriminator, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        self.layers_list = []

        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[0], input_shape=(self.num_input,)))
        bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.layers_list.append(bn)
        activation = tf.keras.layers.LeakyReLU()
        self.layers_list.append(activation)

        for i in range(len(layer_sizes)):
            layer = tf.keras.layers.Dense(layer_sizes[i])
            self.layers_list.append(layer)

            activation = tf.keras.layers.LeakyReLU(0.2)
            self.layers_list.append(activation)

        self.layers_list.append(tf.keras.layers.Dense(num_output))

    def call(self, x, training):
        for i, layer in enumerate(self.layers_list):
            if i == 1 and not training:
                continue
            x = layer(x)
        x = tf.reduce_mean(x)
        return x

    def extract_feature(self, x):
        for i, layer in enumerate(self.layers_list[:5]):
            if i != 1:
                x = layer(x)
        return x

