from typing import List

from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np


class GradientLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def _get_gradients(self, image_batch):
        if len(image_batch.shape) != 5:
            raise Exception("Dimensions not supported")

        resized = tf.reshape(image_batch, (-1, image_batch.shape[2], image_batch.shape[3], image_batch.shape[4]))
        dx, dy = tf.image.image_gradients(resized)
        concatenated = tf.concat((dy, dx), axis=-1)
        return concatenated

    def call(self, y_true, y_pred):
        return tf.norm(self._get_gradients(y_true) - self._get_gradients(y_pred))


class Weighted_MSE(tf.keras.losses.Loss):
    def __init__(self, scaler=3):
        super(Weighted_MSE, self).__init__(reduction='none')
        self.scaler = scaler

    def call(self, y_true, y_pred):
        errors = tf.math.square(tf.math.subtract(y_true, y_pred))
        weights = tf.pow(errors - tf.reduce_min(errors) + 1, self.scaler)
        out = weights * errors

        return tf.math.reduce_mean(out)


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, losses_to_be_combined: List[str]):
        self.losses = [losses[loss] for loss in losses_to_be_combined]

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum([loss(y_true, y_pred) for loss in self.losses])


class GradientWeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weighted_mse_params, gradient_params):
        super(GradientWeightedMSE, self).__init__(reduction='none')
        self.weighted_mse = Weighted_MSE(**weighted_mse_params)
        self.gradient = GradientLoss(**gradient_params)

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum([self.weighted_mse(y_true, y_pred), self.gradient(y_true, y_pred)])


losses = {
    'mse': MeanSquaredError,
    "gradient_loss": GradientLoss,
    "weighted_mse": Weighted_MSE,
    "combined_loss": CombinedLoss,
    "gradient_mse": GradientWeightedMSE
}
