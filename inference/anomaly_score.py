from typing import NamedTuple
from abc import ABC, abstractmethod

import tensorflow as tf


def mse_standarized(input, output):
    abnormality_scores = tf.math.sqrt(tf.math.reduce_sum(
        tf.math.square(input - output), axis=(1, 2, 3, 4)))
    minimum, maximum = tf.math.reduce_min(
        abnormality_scores), tf.math.reduce_max(abnormality_scores)
    abnormality_scores = tf.math.divide(
        tf.math.subtract(abnormality_scores, minimum), maximum)
    return tf.reshape(abnormality_scores, (*abnormality_scores.shape, 1))


def peak_signal_noise_ratio(input, output):
    mse = tf.math.sqrt(
        tf.math.reduce_sum(
            tf.math.square(
                input - output),
            axis=(1, 2, 3, 4)
        )
    )
    maximum = tf.math.reduce_max(mse)
    abnormality_scores = 10 * tf.math.log(
        tf.math.divide(
            maximum,
            mse
        )
    )
    return tf.reshape(abnormality_scores, (*abnormality_scores.shape, 1))


heuristics = {
    'mse': mse_standarized,
    'psnr': peak_signal_noise_ratio
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
