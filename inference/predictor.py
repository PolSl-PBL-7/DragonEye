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
        
    @tf.function
    def calculate_scores(self, dataset, predictions):
        print("calculate scores")
        anomaly_scores = tf.data.Dataset.zip((dataset, predictions)).map(lambda x, y: self.anomaly_score(x, y))
        return anomaly_scores

    def __call__(self, dataset):

        anomaly_scores = None
        print("get predictions")
        predictions = self.reconstruction_model.predict(dataset)
        try:
            dataset = dataset.map(lambda x: x['Input_Dynamic'])
        except Exception:
            print("X is not of type <<class dict>>")

        batch_size = [batch.shape[0] for batch in dataset.take(1)][0]
        predictions = predictions.unbatch().batch(batch_size)
        anomaly_scores = self.calculate_scores(dataset, predictions)
       

        return dataset, predictions, anomaly_scores
