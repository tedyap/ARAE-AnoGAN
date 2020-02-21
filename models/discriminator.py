import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(params.lr_disc)
        self.layer_sizes = params.disc_size

        self.layers_list = []

        self.layers_list.append(tf.keras.layers.Dense(self.layer_sizes[0], input_shape=(params.hidden_size,)))
        if params.batch_norm:
            bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.layers_list.append(bn)

        activation = tf.keras.layers.LeakyReLU(0.2)
        self.layers_list.append(activation)

        for i in range(len(self.layer_sizes)):
            layer = tf.keras.layers.Dense(self.layer_sizes[i])
            self.layers_list.append(layer)

            activation = tf.keras.layers.LeakyReLU(0.2)
            self.layers_list.append(activation)

        self.layers_list.append(tf.keras.layers.Dense(1))

    def call(self, x, training):
        for i, layer in enumerate(self.layers_list):
            if i == 1 and not training:
                continue
            x = layer(x)
        x = tf.reduce_mean(x)
        return x

    def extract_feature(self, x):
        for i, layer in enumerate(self.layers_list[:4]):
            if i != 1:
                x = layer(x)
        return x

