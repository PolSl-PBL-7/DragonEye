from attr import Attribute
import tensorflow as tf
import numpy as np


def get_stochastic_dataset(dataset, force=False):
    shape = ()
    for batch in dataset:
        try:
            shape = batch.shape
        except AttributeError:
            continue

    if len(shape) == 5 or force:
        return dataset.unbatch().batch(1)
    elif len(shape) == 4:
        return dataset.batch(1)
    else:
        raise ValueError("dataset shape must be (batch_size, time_window, width, height, channel), or (time_window, width, height, channel)")


def get_figure_subplot_shape(n_plots: int):
    divisors = np.where(n_plots % np.arange(1, n_plots + 1) == 0)[0] + 1
    n = divisors[int(len(divisors) / 2)]
    m = int(n_plots / n)
    return m, n


def get_dataset_len(dataset):
    return len([0 for _ in dataset])


def min_max_scores(scores):
    min_ = None
    max_ = None
    for batch in scores:
        if min_:
            min_ = {key: batch[key][0].numpy() if batch[key][0].numpy() < min_[key] else min_[key] for key in list(batch.keys())}
        else:
            min_ = {key: batch[key][0].numpy() for key in list(batch.keys())}
        if max_:
            max_ = {key: batch[key][0].numpy() if batch[key][0].numpy() > max_[key] else max_[key] for key in list(batch.keys())}
        else:
            max_ = {key: batch[key][0].numpy() for key in list(batch.keys())}

    return min_, max_
