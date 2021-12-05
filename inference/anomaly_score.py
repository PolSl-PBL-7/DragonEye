from typing import NamedTuple
from abc import ABC, abstractmethod

import tensorflow as tf


def heuristic_anomaly_score(input, output):
    abnormality_scores = tf.math.sqrt(tf.math.reduce_sum(
        tf.math.square(input - output), axis=(1, 2, 3, 4)))
    minimum, maximum = tf.math.reduce_min(
        abnormality_scores), tf.math.reduce_max(abnormality_scores)
    abnormality_scores = tf.math.divide(
        tf.math.subtract(abnormality_scores, minimum), maximum)
    return tf.reshape(abnormality_scores, (1, *abnormality_scores.shape, 1))


class AnomalyScoreConfig(NamedTuple):
    pass


class AnomalyScore(ABC):
    """
    Class responsible for classyfing anomalies based on input video, and reconstructed one.
    """
    @abstractmethod
    def __init__(self, config: AnomalyScoreConfig):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input, output):
        raise NotImplementedError


class AnomalyScoreHeuristic(AnomalyScore):

    def __init__(self, config: AnomalyScoreConfig):
        pass

    def __call__(self, input, output):
        return heuristic_anomaly_score(input, output)
