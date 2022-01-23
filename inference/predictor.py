from typing import NamedTuple
from abc import ABC, abstractmethod
from inference import anomaly_score

from inference.anomaly_score import AnomalyScore, AnomalyScoreHeuristic, AnomalyScoreConfig
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
        predictions = tf.data.Dataset.from_tensors(self.reconstruction_model.predict(dataset)).unbatch()
        try:
            dataset = dataset.map(lambda x: x['Input_Dynamic'])
        except Exception:
            print("X is not of type <<class dict>>")

        batch_size = [batch.shape[0] for batch in dataset.take(1)][0]
        predictions.batch(batch_size)

        for batch, pred in tf.data.Dataset.zip((dataset, predictions)):

            scores = self.anomaly_score(batch, pred)
            if anomaly_scores:
                anomaly_scores = anomaly_scores.concatenate(
                    tf.data.Dataset.from_tensor_slices(scores))
            else:
                anomaly_scores = tf.data.Dataset.from_tensor_slices(scores)

        return anomaly_scores
