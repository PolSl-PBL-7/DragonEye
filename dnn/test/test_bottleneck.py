import tensorflow as tf

from dnn.models.conv.blocks import ChongTayConvLstmBottleneckBlock


def test_conv_lstm_bottleneck_forward_default():
    model = ChongTayConvLstmBottleneckBlock()
    x = tf.random.normal((1, 26, 26, 10, 64))
    out = model(x)
    assert tf.TensorShape(out.shape) == tf.TensorShape((1, 26, 26, 10, 64))
