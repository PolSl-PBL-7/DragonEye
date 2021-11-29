import tensorflow as tf

from dnn.models.conv.blocks import ChongTayDecoder


def test_forward_default():
    model = ChongTayDecoder()
    x = tf.random.normal((1, 10, 26, 26, 64))
    out = model(x)
    assert tf.TensorShape(out.shape) == tf.TensorShape((1, 10, 227, 227, 1))
