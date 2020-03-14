import os
import logging
import tensorflow as tf
import pandas as pd
from models.autoencoder import Seq2Seq
from models.discriminator import Discriminator
from models.generator import Generator
from opts import configure_args
from build_vocab import Corpus
from utils import Params, Metrics, Checkpoints, set_logger, set_seeds
from tensorboardX import SummaryWriter


def compute_anomaly_score(source, step, ano_para=0.1):
    with tf.GradientTape() as tape:
        noise = tf.random.normal((source.shape[0], 100))
        fake_latent = ano_gen(noise)
        real_latent = autoencoder(source, encode_only=True, noise=False)

        fake_feature = discriminator.extract_feature(fake_latent)
        real_feature = discriminator.extract_feature(real_latent)

        anogan_res_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(fake_latent, real_latent))))
        anogan_disc_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(fake_feature, real_feature))))

        anomaly_score = (1 - ano_para) * anogan_res_loss + ano_para * anogan_disc_loss

    gradients = tape.gradient(anomaly_score, ano_gen.trainable_variables)
    tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5).apply_gradients(zip(gradients, ano_gen.trainable_variables))
    tb_writer.add_scalar('train/anomaly_score', anomaly_score.numpy(), step)
    return anomaly_score


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    if not os.path.exists(os.path.join(args.model_dir, 'output')):
        os.makedirs(os.path.join(args.model_dir, 'output'))
    set_logger(os.path.join(args.model_dir, 'output/train.log'))
    set_seeds(args.seed)

    # Prepare dataset
    logging.info('Preparing dataset...')
    corpus = Corpus(args.data_dir, n_tokens=args.vocab_size)
    args.vocab_size = min(args.vocab_size, corpus.vocab_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((corpus.test_source, corpus.test_target, corpus.test_label)).batch(32)

    # Models
    autoencoder = Seq2Seq(params, args)
    discriminator = Discriminator(params)
    generator = Generator(params)

    autoencoder.trainable = False
    discriminator.trainable = False
    generator.trainable = False

    # Optimizers
    ae_optim = tf.keras.optimizers.SGD(params.lr_ae)
    #ae_optim = tf.keras.optimizers.RMSprop(params.lr_ae)
    disc_optim = tf.keras.optimizers.Adam(params.lr_disc, params.beta1)
    gen_optim = tf.keras.optimizers.Adam(params.lr_gen, params.beta1)

    models = autoencoder, discriminator, generator
    optimizers = ae_optim, disc_optim, gen_optim

    tb_writer = SummaryWriter(logdir=os.path.join(args.model_dir, 'anogan'))

    ckpts = Checkpoints(models, optimizers, os.path.join(args.model_dir, 'ckpts'))
    ckpts.restore()

    if ckpts.has_ckpts:
        logging.info("Restored from {}".format(ckpts.has_ckpts))
    else:
        logging.info("ARAE is not trained...")

    z_input = tf.keras.layers.Input(shape=(params.noise_size,))
    g_input = tf.keras.layers.Dense(params.noise_size, trainable=True, activation='sigmoid')(z_input)
    g_output = generator.generate(g_input)
    ano_gen = tf.keras.Model(inputs=z_input, outputs=g_output)
    df_idx = 0
    df = pd.DataFrame(columns=['sentence', 'label', 'loss'])
    logging.info('Label 0 indicates a normal sample and label 1 indicates an anomaly...')
    convert_tokens2sents = lambda tokens: corpus.dictionary.convert_idxs2tokens_prettified(tokens)

    for batch, (source, target, label) in enumerate(test_dataset):
        real_latent = autoencoder(source, encode_only=True, noise=False)
        output = autoencoder.generate(source, real_latent, 15).numpy()
        ge_sentences = [' '.join(convert_tokens2sents(tokens)) for tokens in output]
        for i in ge_sentences:
            print(batch, i)

        """
        for i in range(args.anogan_epoch):
            anomaly_score = compute_anomaly_score(source, i)

        source_sentence = ' '.join(corpus.dictionary.convert_idxs2tokens_prettified(source.numpy()[0]))

        print(source_sentence)

        df = df.append({'sentence': source_sentence, 'label': int(label[0].numpy()), 'loss': anomaly_score.numpy()}, ignore_index=True)
        """
        if batch == 0:
            break


    print(df)
