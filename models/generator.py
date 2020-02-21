import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(params.lr_gen)
        self.noise_size = params.noise_size
        self.layer_sizes = params.gen_size
        self.layers_list = []
        self.layers_list.append(tf.keras.layers.Dense(self.layer_sizes[0], input_shape=(params.noise_size,)))

        if params.batch_norm:
            bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.layers_list.append(bn)

        activation = tf.keras.layers.LeakyReLU(0.2)
        self.layers_list.append(activation)

        for i in range(len(self.layer_sizes)):
            layer = tf.keras.layers.Dense(self.layer_sizes[i])
            self.layers_list.append(layer)
            if params.batch_norm:
                bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
                self.layers_list.append(bn)

            activation = tf.keras.layers.LeakyReLU(0.2)
            self.layers_list.append(activation)

        self.layers_list.append(tf.keras.layers.Dense(params.hidden_size))

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
            layer.trainable = False
            x = layer(x)
        return x

