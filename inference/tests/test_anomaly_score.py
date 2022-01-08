from inference.anomaly_score import heuristics
import tensorflow as tf


def test_heuristic_anomaly_score():
    true = tf.random.normal((32, 10, 227, 227, 3))
    pred = tf.random.normal((32, 10, 227, 227, 3))

    scores = heuristics['mse'](true, pred)

    assert scores.shape, (32, 1)
