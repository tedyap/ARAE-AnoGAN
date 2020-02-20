from opts import configure_args
import logging
import os

import tensorflow as tf
from models.autoencoder import Seq2Seq
from models.discriminator import Discriminator
from models.generator import Generator
from build_vocab import Corpus
from utils import Params


if __name__ == '__main__':
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load data
    corpus = Corpus(args.data_dir, args.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((corpus.train_source, corpus.train_target)).batch(params.batch_size)

    autoencoder = Seq2Seq(params, args)
    discriminator = Discriminator(params)
    generator = Generator(params)

    autoencoder.trainable = True
    discriminator.trainable = True
    generator.trainable = True






