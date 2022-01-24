from typing import NamedTuple
from abc import ABC, abstractmethod

import tensorflow as tf

def mse_standarized(input, output):
    input = input[:,-1,:,:,:]
    output = output[:,-1,:,:,:]
    abnormality_scores = tf.math.reduce_sum(tf.math.abs(input - output), axis=(1,2,3))/(input.shape[1]*input.shape[2]*input.shape[3])
    return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))


def peak_signal_noise_ratio(input, output):
    input = input[:,-1,:,:,:]
    output = output[:,-1,:,:,:]
    mse = tf.math.sqrt(
        tf.math.reduce_sum(
            tf.math.square(
                input - output),
            axis=(1, 2, 3)
        )
    )/input.shape[1]*input.shape[2]*input.shape[3]
    maximum = tf.math.reduce_max(mse)
    abnormality_scores = 10 * tf.math.log(
        tf.math.divide(
            maximum,
            mse
        )
    )
    return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))


heuristics = {
    'mse': mse_standarized,
    'mean squared error': mse_standarized,
    'psnr': peak_signal_noise_ratio,
    'peak signal noise ratio': peak_signal_noise_ratio
}


class AnomalyScoreConfig(NamedTuple):
    metrics: list = ['mse', "psnr"]


class AnomalyScore(ABC):
    """
    Class responsible for classyfing anomalies based on input video, and reconstructed one.
    """
    @abstractmethod
    def __init__(self, config: AnomalyScoreConfig):
        self.config = config

    @abstractmethod
    def __call__(self, input, output):
        raise NotImplementedError


class AnomalyScoreHeuristic(AnomalyScore):

    def __init__(self, config: AnomalyScoreConfig):
        self.config = config

    def __call__(self, input, output):
        return tf.transpose([heuristics[m](input, output) for m in self.config.metrics])
