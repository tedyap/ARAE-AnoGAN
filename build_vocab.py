import os
import logging
import numpy as np
import pandas as pd


class Dictionary:
    """
    Dictionary class to create word-to-index and index-to-word dictory to store vocabulary.

    Reference: github.com/jakezhaojb/ARAE/blob/master/lang/utils.py
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [word for word in self.wordcounts.items()]

        # prune by most frequently seen words
        vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]

        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        logging.info("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus:
    def __init__(self, path, maxlen, vocab_size, lowercase=False):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size - 4
        self.train_path = os.path.join(path, 'reuters_train.txt')
        self.test_path = os.path.join(path, 'reuters_test.txt')
        self.df_test = pd.DataFrame(columns=['sentence', 'label', 'loss'])

        # make the vocabulary from training set
        self.make_vocab()

        self.train_source = self.tokenize(self.train_path, True)
        self.train_target = self.tokenize(self.train_path, False)
        self.test_source = self.tokenize(self.test_path, True, True)
        self.test_target = self.tokenize(self.test_path, False)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size)

    def tokenize(self, path, source, label=False):
        """Tokenizes a text file and returns a list of list of indices of each line in the data."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            wordcount = 0
            lines = []
            for i, line in enumerate(f):
                linecount += 1
                line = line.split(",")
                words = line[0].split()
                if len(words) > self.maxlen - 2:
                    dropped += 1
                    continue
                wordcount = wordcount + len(words)
                # convert word to index
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                # pad with zeroes so length of sentence is equal to max length
                indices = np.pad(indices, [0, self.maxlen - len(indices)], 'constant', constant_values=(0, 0)).tolist()
                if source is True:
                    indices = [1] + indices
                else:
                    indices = indices + [2]
                lines.append(indices)
                if label:
                    self.df_test = self.df_test.append({"sentence": line[0], "label": int(line[1][0]), "loss": 0},
                                                       ignore_index=True)

        np.asarray(lines)

        logging.info("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        logging.info("Average sentence length: {:.2f}".format(wordcount / (linecount - dropped)))
        return lines
