import tensorflow as tf


class Seq2Seq(tf.keras.Model):
    def __init__(self, params, args):
        super(Seq2Seq, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD(params.lr_ae)
        self.embedding = tf.keras.layers.Embedding(args.vocab_size, params.embedding_size, name="Enc-Embed")
        self.embedding_decoder = tf.keras.layers.Embedding(args.vocab_size, params.embedding_size, name="Dec-Embed")
        self.encoder_lstm = tf.keras.layers.LSTM(
            params.hidden_size, return_sequences=True, return_state=True, name="Enc-LSTM")
        self.decoder_lstm = tf.keras.layers.LSTM(
            params.hidden_size, return_sequences=True, return_state=True, name="Dec-LSTM")
        self.dense = tf.keras.layers.Dense(args.vocab_size, name="Dec-Dense")
        self.hidden_size = params.hidden_size
        self.noise_radius = params.noise_radius
        self.training = True

    def init_state(self, batch_size):
        return tf.zeros([1, batch_size, self.hidden_size])

    def encode(self, indices, noise):
        embed = self.embedding(indices)
        output, state_h, state_c = self.encoder_lstm(embed)

        # normalize to unit ball
        state_h = tf.math.l2_normalize(state_h, -1)

        # add gaussian noise
        if noise and self.noise_radius > 0:
            state_h = state_h + tf.random.normal(shape=tf.shape(state_h), mean=0.0, stddev=self.noise_radius,
                                                 dtype=tf.float32)
        return state_h

    def decode(self, indices, hidden, maxlen):
        batch_size = indices.shape[0]
        # (batch_size, hidden_size) -> (batch_size, maxlen, hidden_size)
        all_hidden = tf.tile(tf.expand_dims(hidden, 1), tf.constant([1, maxlen, 1]))

        # state = (tf.expand_dims(hidden, 0), self.init_state(batch_size))
        state = (hidden, tf.zeros([batch_size, self.hidden_size]))

        embeddings = self.embedding_decoder(indices)

        # need to check shape
        augmented_embeddings = tf.concat([embeddings, all_hidden], 2)
        output, state_h, state_c = self.decoder_lstm(augmented_embeddings, state)
        logits = self.dense(output)

        # (batch_size, maxlen, vocab_size)
        return logits

    def generate(self, indices, hidden, maxlen, sample=False):
        batch_size = indices.shape[0]
        state = (hidden, tf.zeros([batch_size, self.hidden_size]))
        start = tf.ones([batch_size, 1])

        embeddings = self.embedding_decoder(start)
        all_hidden = tf.expand_dims(hidden, 1)
        augmented_embeddings = tf.concat([embeddings, all_hidden], 2)
        all_indices = tf.ones([batch_size, 1], dtype=tf.int64)
        output, state_h, state_c = self.decoder_lstm(augmented_embeddings, state)
        logits = self.dense(output)
        for i in range(maxlen - 1):
            output, state_h, state_c = self.decoder_lstm(augmented_embeddings, state)
            logits = self.dense(output)

            if not sample:
                indices = tf.math.argmax(logits, -1)
            else:
                indices = tf.random.categorical(tf.squeeze(logits), num_samples=1)

            all_indices = tf.concat([all_indices, indices], axis=1)
            embeddings = self.embedding_decoder(indices)
            augmented_embeddings = tf.concat([embeddings, all_hidden], 2)
        return all_indices

    def call(self, indices, noise):
        hidden = self.encode(indices, noise)
        decoded = self.decode(indices, hidden, maxlen=31)
        return hidden, decoded

