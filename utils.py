import os
import json
import logging
import tensorflow as tf
from collections import defaultdict


class Params:
    """Class that loads hyperparameters from a json file.
    Reference: github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
    Example:
    ```
    params = Params(json_path)
    logging.info(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class Metrics(defaultdict):
    def __init__(self, **kwargs):
        super().__init__(int)
        self._accum_dict(kwargs)

    def _accum_dict(self, d):
        for key, val in d.items():
            self[key] = self.get(key, 0) + val

    def accum(self, *args, **kwargs):
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Can process only dicts")
            self._accum_dict(arg)
        self._accum_dict(kwargs)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def iget_line(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            yield line.strip()


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


class Checkpoints:
    def __init__(self, models, optimizers, ckpts_dir):
        autoencoder, discriminator, generator = models
        ae_optim, disc_optim, gen_optim = optimizers

        # Checkpoint managers
        self.ae_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=ae_optim, net=autoencoder)
        self.ae_manager = tf.train.CheckpointManager(self.ae_ckpt, os.path.join(ckpts_dir, 'ae_ckpts'),
                                                max_to_keep=1)

        self.disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=disc_optim, net=discriminator)
        self.disc_manager = tf.train.CheckpointManager(self.disc_ckpt, os.path.join(ckpts_dir, 'disc_ckpts'),
                                                  max_to_keep=1)

        self.gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=gen_optim, net=generator)
        self.gen_manager = tf.train.CheckpointManager(self.gen_ckpt, os.path.join(ckpts_dir, 'gen_ckpts'),
                                                 max_to_keep=1)

        self.ae_ckpt.restore(self.ae_manager.latest_checkpoint)
        self.disc_ckpt.restore(self.disc_manager.latest_checkpoint)
        self.gen_ckpt.restore(self.gen_manager.latest_checkpoint)

        self.has_ckpts = self.ae_manager.latest_checkpoint

    def save(self):
        self.ae_manager.save()
        self.disc_manager.save()
        self.gen_manager.save()

    def restore(self):
        self.ae_ckpt.restore(self.ae_manager.latest_checkpoint)
        self.disc_ckpt.restore(self.disc_manager.latest_checkpoint)
        self.gen_ckpt.restore(self.gen_manager.latest_checkpoint)
