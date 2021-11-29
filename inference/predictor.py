from typing import NamedTuple
from abc import ABC, abstractmethod

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
        anomaly_scores = tf.data.Dataset()
        for batch in dataset:
            predictions = self.reconstruction_model.predict(dataset)
            scores = self.anomaly_score(batch, predictions)
            anomaly_scores.concatenate(scores)
        
        return anomaly_scores


