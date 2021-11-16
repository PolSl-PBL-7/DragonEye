import tensorflow as tf

from dnn.models.conv.blocks import ChongTayEncoder


def test_forward_default():
    model = ChongTayEncoder()
    x = tf.random.normal((1, 227, 227, 10, 1))
    out = model(x)
    assert tf.TensorShape(out.shape) == tf.TensorShape((1, 26, 26, 10, 64))
