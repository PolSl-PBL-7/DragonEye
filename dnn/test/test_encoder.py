import tensorflow as tf

from dnn.models.conv.blocks import ChongTayEncoder


def test_forward_default():
    model = ChongTayEncoder()
    x = tf.random.normal((1, 10, 227, 227, 1))
    out = model(x)
    assert tf.TensorShape(out.shape) == tf.TensorShape((1, 10, 26, 26, 64))
