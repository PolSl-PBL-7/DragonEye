import tensorflow as tf

from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, ModelConfig


def test_default_autoencoder():
    config = ModelConfig()
    model = SpatioTemporalAutoencoder(config)
    x = tf.random.normal((1, 10, 227, 227, 1))
    out = model(x)
    assert tf.TensorShape(out.shape) == tf.TensorShape((1, 10, 227, 227, 1))
    assert ((tf.reduce_max(out) <= 1.) & (tf.reduce_min(out) >= 0.))
