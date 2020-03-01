import os
import logging
import tensorflow as tf
from itertools import islice
from models.autoencoder import Seq2Seq
from models.discriminator import Discriminator
from models.generator import Generator
from opts import configure_args
from data import Corpus
from utils import Params, Metrics, Checkpoints, set_logger
import numpy as np
import random
from tensorboardX import SummaryWriter


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def gradient_penalty(models, real_latent, fake_latent, params):
    autoencoder, discriminator, generator = models
    epsilon = tf.random.uniform([real_latent.shape[0], 300])
    x_hat = epsilon * real_latent + (1 - epsilon) * fake_latent
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator(x_hat, training=False)
    gradients = t.gradient(d_hat, x_hat)
    grad_penalty = tf.reduce_mean((tf.math.l2_normalize(gradients, 1) - 1) ** 2)
    return params.grad_lambda * grad_penalty


def generate_sentence(models, source, corpus, step, args):
    autoencoder, discriminator, generator = models
    convert_tokens2sents = lambda tokens: corpus.dictionary.convert_idxs2tokens_prettified(tokens)

    source_lines = source.numpy()
    or_sentences = [' '.join(convert_tokens2sents(tokens)) for tokens in source_lines]

    # generate output from autoencoder
    logits = autoencoder(source, noise=False)
    ae_lines = tf.math.argmax(logits, 2).numpy()
    ae_sentences = [' '.join(convert_tokens2sents(tokens)) for tokens in ae_lines]

    noise = tf.random.normal([source.shape[0], 100])
    fake_latent = generator(noise, training=False)
    output = autoencoder.generate(source, fake_latent, 15, sample=True).numpy()
    ge_sentences = [' '.join(convert_tokens2sents(tokens)) for tokens in output]

    with open(os.path.join(args.model_dir, 'output.txt'), 'w') as file:
        file.write('Step {} \n'.format(step))
        for i in range(len(or_sentences)):
            file.write("OR: " + or_sentences[i] + '\n')
            file.write("AE: " + ae_sentences[i] + '\n')
            file.write("GE: " + ge_sentences[i] + '\n\n')


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    set_seeds(args.seed)

    corpus = Corpus(args.data_dir, n_tokens=args.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((corpus.train_source, corpus.train_target)).batch(params.batch_size)

    # Models
    autoencoder = Seq2Seq(params, args)
    discriminator = Discriminator(params)
    generator = Generator(params)

    autoencoder.trainable = True
    discriminator.trainable = True
    generator.trainable = True

    # Optimizers
    ae_optim = tf.keras.optimizers.SGD(params.lr_ae)
    disc_optim = tf.keras.optimizers.Adam(params.lr_disc, params.beta1)
    gen_optim = tf.keras.optimizers.Adam(params.lr_gen, params.beta1)

    models = autoencoder, discriminator, generator
    optimizers = ae_optim, disc_optim, gen_optim

    ckpts = Checkpoints(models, optimizers, args.ckpts_dir)
    ckpts.restore()

    print(corpus.train_source[:5])
