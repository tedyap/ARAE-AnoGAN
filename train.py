import os
import logging
import tensorflow as tf
from itertools import islice
import random

from models.autoencoder import Seq2Seq
from models.discriminator import Discriminator
from models.generator import Generator
from opts import configure_args
from data import Corpus
from utils import Params, Metrics, Checkpoints, set_logger, static_vars
import numpy as np
from tensorboardX import SummaryWriter


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@static_vars(niter=0)
def form_log_line(metrics):
    metrics['loss_d'] = metrics['disc_loss'] / metrics['niter_clear']
    metrics['loss_g'] = metrics['gen_loss'] / metrics['niter_clear']
    line = '[{epoch:3d}/{nepoch:3d}][{iter:5d}/{niter:5d}] | ' \
           'acc {acc:4.2f} | '\
           'loss_d {loss_d:.3f} | loss_g {loss_g:.3f}'.format(**metrics)
    tb_writer.add_scalar('train/acc', metrics['acc'], form_log_line.niter)
    tb_writer.add_scalar('train/loss_d', metrics['loss_d'], form_log_line.niter)
    tb_writer.add_scalar('train/loss_g', metrics['loss_g'], form_log_line.niter)

    print('{{"metric": "acc", "value": {}, "step": {}}}'.format(float(metrics['acc']), form_log_line.niter))
    print('{{"metric": "loss_d", "value": {}, "step": {}}}'.format(float(metrics['loss_d']), form_log_line.niter))
    print('{{"metric": "loss_g", "value": {}, "step": {}}}'.format(float(metrics['loss_g']), form_log_line.niter))

    form_log_line.niter += 1
    return line


def loss_func(targets, logits):
    # use SparseCat.Crossentropy since targets are not one-hot encoded
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # not take zero (<pad>) target into account when computing the loss since sequence is padded
    mask = tf.math.logical_not(tf.math.equal(targets, 0))

    # accuracy
    idx = tf.cast(tf.math.argmax(logits, 2), dtype=tf.int32)
    match = tf.math.equal(idx, targets)
    logical_and = tf.math.logical_and(match, mask)
    accuracy = tf.reduce_sum(tf.cast(logical_and, dtype=tf.int32)) / (targets.shape[0] * targets.shape[1]) * 100

    # crossentropy loss
    mask = tf.cast(mask, dtype=tf.int32)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss, accuracy


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


def train_autoencoder(models, optimizers, source, target, params):
    autoencoder, discriminator, generator = models
    ae_optim, disc_optim, gen_optim = optimizers
    with tf.GradientTape() as tape:
        logits = autoencoder(source, noise=True)
        ae_loss, accuracy = loss_func(target, logits)

    gradients = tape.gradient(ae_loss, autoencoder.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, params.clip)
    ae_optim.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    return {'ae_loss': ae_loss.numpy(), 'acc': accuracy.numpy()}


def train_disc(models, optimizers, source):
    autoencoder, discriminator, generator = models
    ae_optim, disc_optim, gen_optim = optimizers
    with tf.GradientTape() as disc_tape:
        real_latent = autoencoder(source, encode_only=True, noise=False)
        disc_real_loss = discriminator(real_latent, training=True)
        noise = tf.random.normal((source.shape[0], generator.noise_size))
        fake_latent = generator(noise, training=False)
        disc_fake_loss = discriminator(fake_latent, training=True)
        #grad_penalty = gradient_penalty(models, real_latent, fake_latent, params)
        disc_loss = tf.reduce_mean(disc_real_loss - disc_fake_loss)
        #disc_loss_grad_penalty = disc_loss + grad_penalty
        disc_loss_grad_penalty = disc_loss

    disc_gradients = disc_tape.gradient(disc_loss_grad_penalty, discriminator.trainable_variables)
    disc_optim.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    metrics = {'disc_loss': disc_loss.numpy()}
    return metrics


def train_encoder_by_disc(models, optimizers, source, params):
    autoencoder, discriminator, generator = models
    ae_optim, disc_optim, gen_optim = optimizers
    # only train encoder
    for layer_name in ['Dec-Embed', 'Dec-LSTM', 'Dec-Dense']:
        autoencoder.get_layer(layer_name).trainable = False

    """
    @tf.custom_gradient
    def autoencoder_grad_norm(x):
        def grad(dy):
            return 0.1 * dy

        return tf.identity(x), grad
    """

    with tf.GradientTape() as auto_tape:
        real_latent = autoencoder(source, encode_only=True, noise=False)
        #real_latent = autoencoder_grad_norm(real_latent)
        enc_loss = - tf.reduce_mean(discriminator(real_latent, training=False))

    enc_gradients = auto_tape.gradient(enc_loss, autoencoder.trainable_variables)
    # prevent exploding gradient of LSTM
    enc_gradients, _ = tf.clip_by_global_norm(enc_gradients, params.clip)
    ae_optim.apply_gradients(zip(enc_gradients, autoencoder.trainable_variables))

    for layer_name in ['Dec-Embed', 'Dec-LSTM', 'Dec-Dense']:
        autoencoder.get_layer(layer_name).trainable = True

    return {'enc_loss': enc_loss.numpy()}


def train_gen(models, optimizers, source):
    autoencoder, discriminator, generator = models
    ae_optim, disc_optim, gen_optim = optimizers
    with tf.GradientTape() as tape:
        noise = tf.random.normal((source.shape[0], generator.noise_size))
        fake_latent = generator(noise, training=True)
        gen_loss = tf.reduce_mean(discriminator(fake_latent, training=False))

    gradients = tape.gradient(gen_loss, generator.trainable_variables)
    gen_optim.apply_gradients(zip(gradients, generator.variables))

    return {'gen_loss': gen_loss.numpy()}


def train(models, optimizers, dataset, corpus, ckpts, params, args):
    epoch_num = params.epoch_num
    epoch_gan = params.epoch_gan
    batch_epoch = params.batch_epoch
    autoencoder.noise_radius = params.noise_radius
    step = 0

    for e in range(epoch_num, params.max_epoch):
        for batch, (source, target) in islice(enumerate(dataset), batch_epoch, None):
            metrics = Metrics(
                epoch=e,
                max_epoch=params.max_epoch,
            )
            for p in range(params.epoch_ae):
                ae_metrics = train_autoencoder(models, optimizers, source, target, params)
                metrics.accum(ae_metrics)
            for q in range(params.epoch_gan):
                for r in range(params.epoch_disc):
                    disc_metrics = train_disc(models, optimizers, source)
                    metrics.accum(disc_metrics)
                for r in range(params.epoch_enc):
                    enc_metrics = train_encoder_by_disc(models, optimizers, source, params)
                    metrics.accum(enc_metrics)
                for t in range(params.epoch_gen):
                    gen_metrics = train_gen(models, optimizers, source)
                    metrics.accum(gen_metrics)

            batch_epoch += 1
            # anneal noise every 10 batch_epoch for now
            if batch_epoch % 10 == 0:
                autoencoder.noise_radius = autoencoder.noise_radius * 0.9995
            if batch_epoch % 1 == 0:
                #ckpts.save()
                logging.info('--- Epoch {}/{} Batch {} ---'.format(e + 1, metrics['max_epoch'], batch_epoch))
                logging.info('Loss {:.4f}'.format(float(metrics['ae_loss'])))
                logging.info('Disc_Loss {:.4f}'.format(float(metrics['disc_loss'])))
                logging.info('Gen_Loss {:.4f}'.format(float(metrics['gen_loss'])))

                #params.batch_epoch = batch_epoch
                #params.epoch_num = epoch_num
                #params.epoch_gan = epoch_gan
                #params.noise_radius = autoencoder.noise_radius
                #params.save(os.path.join(args.model_dir, 'params.json'))

                # Floydhub metrics
                print('{{"metric": "acc", "value": {}, "step": {}}}'.format(float(metrics['acc']), step))
                print('{{"metric": "ae_loss", "value": {}, "step": {}}}'.format(float(metrics['ae_loss']), step))
                print('{{"metric": "disc_loss", "value": {}, "step": {}}}'.format(float(metrics['disc_loss']), step))
                print('{{"metric": "gen_loss", "value": {}, "step": {}}}'.format(float(metrics['gen_loss']), step))

                step += 1
                tb_writer.add_scalar('train/ae_loss', metrics['ae_loss'], step)
                tb_writer.add_scalar('train/disc_loss', metrics['disc_loss'], step)
                tb_writer.add_scalar('train/gen_loss', metrics['gen_loss'], step)

                generate_sentence(models, source, corpus, step, args)
        batch_epoch = 0


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

    with open(os.path.join(args.model_dir, 'output/output.txt'), 'a+') as file:
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

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'output/train.log'))
    set_seeds(args.seed)

    logging.info('Preparing dataset...')
    corpus = Corpus(args.data_dir, n_tokens=args.vocab_size)
    args.vocab_size = min(args.vocab_size, corpus.vocab_size)
    print(args.vocab_size)
    print(len(corpus.train_source))

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

    tb_writer = SummaryWriter(logdir='summary')

    ckpts = Checkpoints(models, optimizers, args.ckpts_dir)
    ckpts.restore()

    if ckpts.has_ckpts:
        logging.info("Restored from {}".format(ckpts.has_ckpts))
    else:
        logging.info("Initializing from scratch...")

    logging.info('Training...')
    train(models, optimizers, dataset, corpus, ckpts, params, args)













