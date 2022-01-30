from typing import NamedTuple
from abc import ABC, abstractmethod

import tensorflow as tf


def mse(input, output):
    input = input[:, -1, :, :, :]
    output = output[:, -1, :, :, :]
    mae = tf.math.abs(input - output)
    abnormality_scores = tf.math.reduce_sum(mae, axis=(1, 2, 3)) / (input.shape[1] * input.shape[2] * input.shape[3])
    return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))


def peak_signal_noise_ratio(input, output):
    input = input[:, -1, :, :, :]
    output = output[:, -1, :, :, :]
    mse = tf.math.sqrt(
        tf.math.reduce_sum(
            tf.math.square(
                input - output),
            axis=(1, 2, 3)
        )
    ) / input.shape[1] * input.shape[2] * input.shape[3]
    maximum = tf.math.reduce_max(mse)
    abnormality_scores = 10 * tf.math.log(
        tf.math.divide(
            maximum,
            mse
        )
    )
    return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))


def ssim(input, output, c1=0.01, c2=0.03):
    input = input[:, -1, :, :, :]
    output = output[:, -1, :, :, :]
    in_mean = tf.reduce_mean(input, axis=(1, 2, 3))
    out_mean = tf.reduce_mean(output, axis=(1, 2, 3))
    in_variance = tf.math.pow(tf.math.reduce_std(input, axis=(1, 2, 3)), 2)
    out_variance = tf.math.pow(tf.math.reduce_std(output, axis=(1, 2, 3)), 2)
    covariance = tf.multiply(
        tf.reduce_mean(input - tf.reduce_mean(input, axis=(1, 2, 3)), axis=(1, 2, 3)),
        tf.reduce_mean(output - tf.reduce_mean(output, axis=(1, 2, 3)), axis=(1, 2, 3)),
    )
    numerator = tf.multiply(
        2 * tf.multiply(in_mean, out_mean) + c1,
        2 * covariance + c2,
    )
    denominator = tf.multiply(
        tf.add(
            tf.math.pow(in_mean, 2),
            tf.math.pow(out_mean, 2),
        ) + c1,
        tf.add(
            tf.math.pow(in_variance, 2),
            tf.math.pow(out_variance, 2),
        ) + c2

    )
    metric = tf.math.divide(numerator, denominator)
    print(metric, metric.shape)
    return tf.reshape(metric, [-1, 1])


heuristics = {
    'mse': mse,
    'mean squared error': mse,
    'psnr': peak_signal_noise_ratio,
    'peak signal noise ratio': peak_signal_noise_ratio,
    'ssim': ssim
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
        metrics = tf.transpose([heuristics[m](input, output) for m in self.config.metrics])
        return tf.reshape(metrics, (-1, metrics.shape[-1]))
