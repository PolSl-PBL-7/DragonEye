from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from typing import NamedTuple
from abc import ABC, abstractmethod

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


class Metric(ABC):
    def __init__(self):
        raise NotImplementedError()
    
    def __call__(self, y_true, y_pred):
        raise NotImplementedError()


def make_windows_mse_model_metric(image_shape = (256,256), window_size_part_vertical = 0.05, window_size_part_horizontal = 0.05):
    
    window_size_part_horizontal = float(window_size_part_horizontal)
    window_size_part_vertical = float(window_size_part_vertical)
    window_length_horizontal = int(image_shape[0] * window_size_part_horizontal)
    window_length_vertical = int(image_shape[1] * window_size_part_vertical)
    
    kernel_size= (window_length_horizontal, window_length_vertical)
    strides= (window_length_horizontal, window_length_vertical)
    
    input_true = Input(shape = image_shape)
    input_pred = Input(shape = image_shape)
    
    conv_true = tf.keras.layers.Conv2D(1, kernel_size, strides, padding = 'same', use_bias = False, initializer = MeanInitializer)(input_true)
    conv_pred = tf.keras.layers.Conv2D(1, kernel_size, strides, padding = 'same', use_bias = False, initializer = NegativeMeanInitializer)(input_pred)
    
    subtracted = keras.layers.Subtract()([conv_true, conv_pred])
        
    model = Model(inputs = [input_true, input_pred], outputs=subtracted)
    
    return model
    

def window_mse(y_true, y_pred):
    
    model = make_windows_mse_model_metric()
    return model(y_true, y_pred)
    
class WindowMSE(Metric):
    def __init__(self, image_shape = (256,256), window_size_part_vertical = 0.05, window_size_part_horizontal = 0.05, scaler=None):
        self.window_size_part_horizontal = float(window_size_part_horizontal)
        self.window_size_part_vertical = float(window_size_part_vertical)
        self.window_length_horizontal = int(image_shape[0] * window_size_part_horizontal)
        self.window_length_vertical = int(image_shape[1] * window_size_part_vertical)
        
#         print(".")
        kernel_size= (self.window_length_horizontal, self.window_length_vertical)
        self.strides= (self.window_length_horizontal, self.window_length_vertical)
        self.padding='SAME'
        
        numerator = tf.ones(tf.convert_to_tensor((1, kernel_size[0], kernel_size[1])), dtype=tf.float32)
        denominator = tf.math.reduce_prod(tf.convert_to_tensor(kernel_size))
        denominator = tf.cast(denominator, dtype=tf.float32)
        self.kernel_positive = numerator/denominator
        self.kernel_negative = -1 * self.kernel_positive
        
        self.left_conv = tf.keras.layers.Conv2D(1, kernel_size, self.strides, padding = 'same', use_bias = False).set_weights(self.kernel_positive)
        # print(self.left_conv.get_weights().shape)
        self.right_conv = tf.keras.layers.Conv2D(1, kernel_size, self.strides, padding = 'same', use_bias = False).set_weights(self.kernel_negative)

    def __call__(self, y_true, y_pred):
        
        left = self.left_conv(y_true)
        right = self.right_conv(y_pred)
        out = tf.math.square(tf.math.subtract(left, right), dtype=tf.float32)

        return out
    

    
class MSE(Metric):
    
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred):
        y_true = y_true[:, -1, :, :, :]
        y_pred = y_pred[:, -1, :, :, :]
        mae = tf.math.abs(y_true - y_pred)
        abnormality_scores = tf.math.reduce_sum(mae, axis=(1, 2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])
        return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))
    
class PeakSignalNoiseRatio(Metric):
    
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred):
        y_true = y_true[:, -1, :, :, :]
        y_pred = y_pred[:, -1, :, :, :]
        mse = tf.math.sqrt(
            tf.math.reduce_sum(
                tf.math.square(
                    y_true - y_pred),
                axis=(1, 2, 3)
            )
        ) / y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
        maximum = tf.math.reduce_max(mse)
        abnormality_scores = 10 * tf.math.log(
            tf.math.divide(
                maximum,
                mse
            )
        )
        return tf.reshape(abnormality_scores, (abnormality_scores.shape[0], 1))


    
class SSIM(Metric):
    
    def __init__(self, c1=0.01, c2=0.03):
        self.c1 = c1
        self.c2 = c2
    
    def __call__(self, y_true, y_pred):
        
        y_true = y_true[:, -1, :, :, :]
        y_pred = y_pred[:, -1, :, :, :]
        in_mean = tf.reduce_mean(y_true, axis=(1, 2, 3))
        out_mean = tf.reduce_mean(y_pred, axis=(1, 2, 3))
        in_variance = tf.math.pow(tf.math.reduce_std(y_true, axis=(1, 2, 3)), 2)
        out_variance = tf.math.pow(tf.math.reduce_std(y_pred, axis=(1, 2, 3)), 2)
        covariance = tf.multiply(
            tf.reduce_mean(y_true - tf.reduce_mean(y_true, axis=(1, 2, 3)), axis=(1, 2, 3)),
            tf.reduce_mean(y_pred - tf.reduce_mean(y_pred, axis=(1, 2, 3)), axis=(1, 2, 3)),
        )
        numerator = tf.multiply(
            2 * tf.multiply(in_mean, out_mean) + self.c1,
            2 * covariance + self.c2
        )
        denominator = tf.multiply(
            tf.add(
                tf.math.pow(in_mean, 2),
                tf.math.pow(out_mean, 2),
            ) + self.c1,
            tf.add(
                tf.math.pow(in_variance, 2),
                tf.math.pow(out_variance, 2),
            ) + self.c2

        )
        metric = tf.math.divide(numerator, denominator)
        print(metric, metric.shape)
        return tf.reshape(metric, [-1, 1])


heuristics = {
    'mse': MSE,
    'mean squared error': MSE,
    'psnr': PeakSignalNoiseRatio,
    'peak signal noise ratio': PeakSignalNoiseRatio,
    'ssim': SSIM,
    'window_mse': WindowMSE
}


class AnomalyScoreConfig(NamedTuple):
    metrics: dict = {'mse': {}, "ssim": {}, 'window_mse': {"image_shape": [227, 227]}}


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

    def __call__(self, y_true, y_pred):
        metrics = {m: heuristics[m](**params)(y_true, y_pred) for m, params in self.config.metrics.items()}
        return metrics
