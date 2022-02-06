from tensorflow.keras.metrics import MeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, MeanAbsoluteError
import tensorflow as tf


class MeanInitializer(tf.keras.initializers.Initializer):
    def __init__(self):
      self.initializer = tf.keras.initializers.Ones()
    def __call__(self, shape, dtype=None, **kwargs):
      prod = tf.math.reduce_prod(shape)
      values = self.initializer(shape=shape) / tf.cast(prod, tf.float32)
      return values

class NegativeMeanInitializer(tf.keras.initializers.Initializer):
    def __init__(self):
      self.initializer = MeanInitializer()
    def __call__(self, shape, dtype=None, **kwargs):
      values = -1 * self.initializer(shape=shape)
      return values

def mean_sliding_window_mse(original_image_batch, reconstruction_batch):
  if len(original_image_batch.shape) != 5:
    raise Exception("Dimensions not supported")


metrics = {
    'mse': MeanSquaredError,
    'mape': MeanAbsolutePercentageError,
    'msle': MeanSquaredLogarithmicError,
    'mae': MeanAbsoluteError,
}
