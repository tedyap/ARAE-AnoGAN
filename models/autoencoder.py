import os
import tensorflow as tf
from build_vocab import Corpus
from models.generator import Generator
from opts import configure_args
from utils import Params


class Seq2Seq(tf.keras.Model):
    def __init__(self, params, args):
        super(Seq2Seq, self).__init__()
        self.embedding_encoder = tf.keras.layers.Embedding(
            args.vocab_size,
            params.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            mask_zero=True,
            name="Enc-Embed")
        self.embedding_decoder = tf.keras.layers.Embedding(
            args.vocab_size, params.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            mask_zero=True,
            name="Dec-Embed")
        self.encoder_lstm = tf.keras.layers.LSTM(
            params.hidden_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            recurrent_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            use_bias=False,
            name="Enc-LSTM")
        self.decoder_lstm = tf.keras.layers.LSTM(
            params.hidden_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            recurrent_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            use_bias=False,
            name="Dec-LSTM")
        self.dense = tf.keras.layers.Dense(
            args.vocab_size,
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="Dec-Dense")
        self.hidden_size = params.hidden_size
        self.noise_radius = params.noise_radius
        self.training = True

    def encode(self, indices, noise):
        # (batch_size, max_len, embedding_size)
        embed = self.embedding_encoder(indices)
        mask = self.embedding_encoder.compute_mask(indices)

        # state_h: (batch_size, hidden_size)
        output, state_h, state_c = self.encoder_lstm(embed, mask=mask)

        # normalize to unit ball
        state_h = tf.math.l2_normalize(state_h, -1)

        # add gaussian noise
        if noise and self.noise_radius > 0:
            state_h = state_h + tf.random.normal(shape=tf.shape(state_h), mean=0.0, stddev=self.noise_radius,
                                                 dtype=tf.float32)
        return state_h

    def decode(self, indices, hidden):
        max_len = indices.shape[1]
        # (batch_size, hidden_size) -> (batch_size, max_len, hidden_size)
        all_hidden = tf.tile(tf.expand_dims(hidden, 1), tf.constant([1, max_len, 1]))

        # (batch_size, max_len, embedding_size)
        embeddings = self.embedding_decoder(indices)

        # (batch_size, max_len, hidden_size + embedding_size)
        augmented_embeddings = tf.concat([embeddings, all_hidden], 2)
        mask = self.embedding_encoder.compute_mask(indices)

        # output: (batch_size, max_len, hidden_size)
        output, state_h, state_c = self.decoder_lstm(augmented_embeddings, mask=mask)

        # (batch_size, max_len, vocab_size)
        logits = self.dense(output)

        return logits

    def generate(self, indices, hidden, max_len, sample=True):
        batch_size = indices.shape[0]
        generated_idx = tf.ones([batch_size, 1], dtype=tf.int32)

        # set <sos> as first token
        start_idx = tf.ones([batch_size, 1], dtype=tf.int32)

        all_hidden = tf.expand_dims(hidden, 1)

        state_h = tf.zeros([batch_size, self.hidden_size])
        state_c = tf.zeros([batch_size, self.hidden_size])

        for i in range(max_len - 1):
            # (batch_size, 1, embedding_size)
            embeddings = self.embedding_decoder(start_idx)

            # (batch_size, 1, embedding_size + hidden_size)
            augmented_embeddings = tf.concat([embeddings, all_hidden], 2)

            # (batch_size, 1, hidden_size)
            output, state_h, state_c = self.decoder_lstm(augmented_embeddings, initial_state=[state_h, state_c])

            # (batch_size, 1, vocab_size)
            logits = self.dense(output)

            if not sample:
                idx = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                #logits = tf.nn.softmax(logits, axis=-1)
                idx = tf.random.categorical(tf.squeeze(logits, axis=1), num_samples=1)

            idx = tf.cast(idx, dtype=tf.int32)

            generated_idx = tf.concat([generated_idx, idx], axis=1)

        return generated_idx

    def call(self, indices, encode_only=False, noise=False):
        hidden = self.encode(indices, noise)
        if encode_only:
            return hidden
        decoded = self.decode(indices, hidden)
        return decoded


if __name__ == '__main__':

    tf.random.set_seed(41)
    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    autoencoder = Seq2Seq(params, args)
    generator = Generator(params)

    corpus = Corpus(args.data_dir, n_tokens=args.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((corpus.train_source, corpus.train_target)).batch(params.batch_size)
    batch = next(iter(dataset))
    source = batch[0]
    target = batch[1]
    autoencoder(source)
