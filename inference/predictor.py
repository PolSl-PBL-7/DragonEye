from typing import NamedTuple
from abc import ABC, abstractmethod

from inference import anomaly_score
from inference.anomaly_score import AnomalyScore, AnomalyScoreHeuristic, AnomalyScoreConfig
from utils.report_utils import get_stochastic_dataset
import tensorflow as tf


class PredictorConfig(NamedTuple):
    reconstruction_model: tf.keras.Model
    anomaly_score: AnomalyScore


class Predictor:
    """
    Module responsible for predictions of anomalies on given dataset, using on trained models.
    """
    reconstruction_model: tf.keras.Model
    anomaly_score: AnomalyScore

    def __init__(self, config: PredictorConfig):
        self.reconstruction_model = config.reconstruction_model
        self.anomaly_score = config.anomaly_score

    def __call__(self, dataset):

        anomaly_scores = None
        predictions = None
        for batch in dataset:
            batch_predictions = self.reconstruction_model.predict(batch)

            scores = self.anomaly_score(batch, batch_predictions)
            if anomaly_scores and predictions:
                anomaly_scores = anomaly_scores.concatenate(
                    tf.data.Dataset.from_tensor_slices(scores))
                predictions = predictions.concatenate(
                    tf.data.Dataset.from_tensor_slices(batch_predictions))
            else:
                anomaly_scores = tf.data.Dataset.from_tensor_slices(scores)
                predictions = tf.data.Dataset.from_tensor_slices(batch_predictions)

        return anomaly_scores.map(lambda x: {self.anomaly_score.config.metrics[i]: x[:, i] for i in range(x.shape[-1])}), predictions
