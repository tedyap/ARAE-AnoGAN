import tensorflow as tf
from opts import configure_args
from utils import Params
import os


class Generator(tf.keras.Model):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.noise_size = params.noise_size
        self.layer_sizes = params.gen_size
        self.layers_list = []

        self.layers_list.append(tf.keras.layers.Dense(self.layer_sizes[0],
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(
                                                          stddev=0.02),
                                                      input_shape=(params.noise_size,)))

        activation = tf.keras.layers.ReLU()
        self.layers_list.append(activation)

        for i in range(1, len(self.layer_sizes)):
            layer = tf.keras.layers.Dense(self.layer_sizes[i],
                                          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
            self.layers_list.append(layer)

            if params.batch_norm:
                bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1,
                                                        gamma_initializer=tf.keras.initializers.RandomNormal(
                                                            stddev=0.02))
                self.layers_list.append(bn)

            activation = tf.keras.layers.ReLU()
            self.layers_list.append(activation)

        self.layers_list.append(
            tf.keras.layers.Dense(params.hidden_size,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))

    def call(self, x, training):
        for i, layer in enumerate(self.layers_list):
            if 'batch_normalization' in layer.name and not training:
                layer.trainable = False
            elif 'batch_normalization' in layer.name and training:
                layer.trainable = True
            x = layer(x)
        return x

    def generate(self, x):
        for i, layer in enumerate(self.layers_list[:-1]):
            x = layer(x)
        return x


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    generator = Generator(params)
    x = tf.random.normal([8, 100])
    generator(x, training=True)
    print(generator.summary())
    print(generator.weights)

