import os
import logging
import tensorflow as tf
from itertools import islice
from models.autoencoder import Seq2Seq
from models.discriminator import Discriminator
from models.generator import Generator
from opts import configure_args
from build_vocab import Corpus
from utils import Params, Metrics, Checkpoints, set_logger, generate_output
from tensorboardX import SummaryWriter


def loss_func(targets, logits):
    # use SparseCat.Crossentropy since targets are not one-hot encoded
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # not take zero target into account when computing the loss since sequence is padded
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


def gradient_penalty(models, real_latent, fake_latent, grad_lambda=1):
    autoencoder, discriminator, generator = models
    epsilon = tf.random.uniform([real_latent.shape[0], 300])
    x_hat = epsilon * real_latent + (1 - epsilon) * fake_latent
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator(x_hat, training=True)
    gradients = t.gradient(d_hat, x_hat)
    grad_penalty = tf.reduce_mean((tf.math.l2_normalize(gradients, 1) - 1) ** 2)
    return grad_lambda * grad_penalty


def train_autoencoder(models, source, target):
    autoencoder, discriminator, generator = models
    with tf.GradientTape() as tape:
        _, logits = autoencoder(source, noise=True)

        ae_loss = loss_func(target, logits)

    gradients = tape.gradient(ae_loss, autoencoder.trainable_variables)
    gradients = [(tf.clip_by_norm(grad, 1)) for grad in gradients]
    autoencoder.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    return {'ae_loss': ae_loss.numpy()}


def train_disc(models, source):
    autoencoder, discriminator, generator = models
    with tf.GradientTape() as disc_tape:
        real_latent, _ = autoencoder(source, False)
        disc_real_loss = discriminator(real_latent, training=True)
        noise = tf.random.normal((source.shape[0], generator.noise_size))
        fake_latent = generator(noise, training=True)
        disc_fake_loss = discriminator(fake_latent, training=True)
        grad_penalty = gradient_penalty(models, real_latent, fake_latent)
        disc_loss = disc_real_loss - disc_fake_loss
        disc_loss_grad_penalty = disc_loss + grad_penalty

    disc_gradients = disc_tape.gradient(disc_loss_grad_penalty, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    metrics = {
        'disc_loss': - disc_loss.numpy(),
        'disc_fake_loss': disc_fake_loss.numpy(),
        'disc_real_loss': disc_real_loss.numpy(),
    }
    return metrics


def train_encoder_by_disc(models, source):
    autoencoder, discriminator, generator = models
    # only train encoder
    for layer_name in ['Dec-Embed', 'Dec-LSTM', 'Dec-Dense']:
        autoencoder.get_layer(layer_name).trainable = False

    @tf.custom_gradient
    def autoencoder_grad_norm(x):
        def grad(dy):
            return 0.1 * dy

        return tf.identity(x), grad

    with tf.GradientTape() as auto_tape:
        real_latent, _ = autoencoder(source, False)
        real_latent = autoencoder_grad_norm(real_latent)
        enc_loss = - discriminator(real_latent, training=True)

    enc_gradients = auto_tape.gradient(enc_loss, autoencoder.trainable_variables)
    # prevent exploding gradient of LSTM
    enc_gradients = [(tf.clip_by_norm(grad, 1)) for grad in enc_gradients]
    autoencoder.optimizer.apply_gradients(zip(enc_gradients, autoencoder.trainable_variables))

    for layer_name in ['Dec-Embed', 'Dec-LSTM', 'Dec-Dense']:
        autoencoder.get_layer(layer_name).trainable = True

    return {'enc_loss': enc_loss.numpy()}


def train_gen(models, source):
    autoencoder, discriminator, generator = models
    with tf.GradientTape() as tape:
        noise = tf.random.normal((source.shape[0], generator.noise_size))
        fake_latent = generator(noise, training=True)
        gen_loss = discriminator(fake_latent, training=True)

    gradients = tape.gradient(gen_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradients, generator.variables))

    return {'gen_loss': gen_loss.numpy()}


def train(models, dataset, corpus, ckpts, params, args):
    autoencoder, discriminator, generator = models

    epoch_num = params.epoch_num
    epoch_gan = params.epoch_gan
    batch_epoch = params.batch_epoch
    autoencoder.noise_radius = params.noise_radius
    step = 0

    for e in range(epoch_num, params.max_epoch):
        if e in [2, 4, 6]:
            epoch_gan += 1

        for batch, (source, target) in islice(enumerate(dataset), batch_epoch, None):
            metrics = Metrics(
                epoch=e,
                max_epoch=params.max_epoch,
            )
            for p in range(params.epoch_ae):
                ae_metrics = train_autoencoder(models, source, target)
                metrics.accum(ae_metrics)
            for q in range(params.epoch_gan):
                for r in range(params.epoch_disc):
                    disc_metrics = train_disc(models, source)
                    metrics.accum(disc_metrics)
                for r in range(params.epoch_enc):
                    enc_metrics = train_encoder_by_disc(models, source)
                    metrics.accum(enc_metrics)
                for t in range(params.epoch_gen):
                    gen_metrics = train_gen(models, source)
                    metrics.accum(gen_metrics)

            step += 1
            tb_writer.add_scalar('train/ae_loss', metrics['ae_loss'], step)
            tb_writer.add_scalar('train/disc_loss', metrics['disc_loss'], step)
            tb_writer.add_scalar('train/gen_loss', metrics['gen_loss'], step)

            # Floydhub metrics
            print('{{"metric": "ae_loss", "value": {}}}'.format(float(metrics['ae_loss'])))
            print('{{"metric": "disc_loss", "value": {}}}'.format(float(metrics['disc_loss'])))
            print('{{"metric": "disc_real_loss", "value": {}}}'.format(float(metrics['disc_real_loss'])))
            print('{{"metric": "disc_fake_loss", "value": {}}}'.format(float(metrics['disc_fake_loss'])))
            print('{{"metric": "enc_loss", "value": {}}}'.format(float(metrics['enc_loss'])))
            print('{{"metric": "gen_loss", "value": {}}}'.format(float(metrics['gen_loss'])))

            batch_epoch += 1
            # anneal noise every 10 batch_epoch for now
            if batch_epoch % 10 == 0:
                autoencoder.noise_radius = autoencoder.noise_radius * 0.995
            if batch_epoch % 10 == 0:
                ckpts.save()
                logging.info('--- Epoch {}/{} Batch {} ---'.format(e + 1, metrics['max_epoch'], batch_epoch))
                logging.info('Loss {:.4f}'.format(float(metrics['ae_loss'])))
                logging.info('Disc_Loss {:.4f}'.format(float(metrics['disc_loss'])))
                logging.info('Disc_Real_Loss {:.4f}'.format(float(metrics['disc_real_loss'])))
                logging.info('Disc_Fake_Loss {:.4f}'.format(float(metrics['disc_fake_loss'])))
                logging.info('Gen_Loss {:.4f}'.format(float(metrics['gen_loss'])))
                logging.info('Enc_loss {:.4f}'.format(float(metrics['enc_loss'])))

                generate_output(models, source, corpus, args)
                params.batch_epoch = batch_epoch
                params.epoch_num = epoch_num
                params.epoch_gan = epoch_gan
                params.noise_radius = autoencoder.noise_radius
        batch_epoch = 0


if __name__ == '__main__':
    tf.random.set_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load data
    corpus = Corpus(args.data_dir, args.max_len, args.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((corpus.train_source, corpus.train_target)).batch(params.batch_size)

    autoencoder = Seq2Seq(params, args)
    discriminator = Discriminator(params)
    generator = Generator(params)

    autoencoder.trainable = True
    discriminator.trainable = True
    generator.trainable = True

    models = (autoencoder, discriminator, generator)

    tb_writer = SummaryWriter(logdir=os.path.join(args.model_dir, 'summary'))

    ckpts = Checkpoints(models, args.ckpts_dir)

    if ckpts.has_ckpts:
        logging.info("Restored from {}".format(ckpts.has_ckpts))
    else:
        logging.info("Initializing from scratch.")

    train(models, dataset, corpus, ckpts, params, args)











