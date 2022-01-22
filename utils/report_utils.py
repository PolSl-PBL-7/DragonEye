from attr import Attribute
import tensorflow as tf
import numpy as np

def get_stochastic_dataset(dataset, force = False):
    shape = ()
    for batch in dataset:
        try:
            shape = batch.shape
        except AttributeError:
            continue
    
    if len(shape) == 5  or force:
        return dataset.unbatch().batch(1)
    elif len(shape) == 4:
        return dataset.batch(1)
    else:
        raise ValueError("dataset shape must be (batch_size, time_window, width, height, channel), or (time_window, width, height, channel)")


def get_figure_subplot_shape(n_plots: int):
    divisors = np.where(n_plots % np.arange(1, n_plots + 1) == 0)[0] + 1
    n = divisors[int(len(divisors)/2)]
    m = int(n_plots/n)
    return m, n

def get_dataset_len(dataset):
    i = 0
    for batch in dataset:
        i+=1
    return i

def min_max_scores(scores):
    min = None
    max = None
    for batch in scores:
        if min:
            min = {key: batch[key][0].numpy() if batch[key][0].numpy() < min[key] else min[key] for key in list(batch.keys())}
        else:
            min = {key: batch[key][0].numpy() for key in list(batch.keys())}
        if max:
            max = {key: batch[key][0].numpy() if batch[key][0].numpy() > max[key] else max[key] for key in list(batch.keys())}
        else:
            max = {key: batch[key][0].numpy() for key in list(batch.keys())}
    
    return min, max