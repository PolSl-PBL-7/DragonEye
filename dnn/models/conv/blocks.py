from typing import Sequence, Callable, Optional, List, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization, Dropout, ConvLSTM2D, Permute, Conv3DTranspose

from utils.iteration_utils import check_equal_lengths


class ChongTayEncoder(keras.layers.Layer):
    """
    Encoder from https://arxiv.org/abs/1701.01546 - Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
    """

    def __init__(self, filter_sizes: Sequence = (11, 5), n_filters: Sequence = (128, 64), strides: Sequence = (4, 2),
                 dropout: float = 0., add_batchnorm: bool = False,
                 activation: Callable = tf.nn.relu):
        """

        Parameters
        ----------
        filter_sizes: sequence of conv filter sizes
        n_filters: sequence of filter numbers for all subsequent layers
        strides: sequence of stride values in x/y dimentions for subsequent conv layers
        dropout: amount of dropout for all encoder layers
        add_batchnorm: whether to add batchnorm after all encoder layers
        activation: tensorflow activation function callable
        """
        if not check_equal_lengths(filter_sizes, n_filters, strides):
            raise Exception("All iterable arguments should be of equal length")
        super(ChongTayEncoder, self).__init__()
        self.conv_layers = [Conv3D(filters=n, kernel_size=(ks, ks, 1), strides=(stride, stride, 1))
                            for n, ks, stride in zip(n_filters, filter_sizes, strides)]
        self.activation = activation

        self.bn: Optional[List[Any]] = None
        if add_batchnorm:
            self.bn = [BatchNormalization() for i in range(len(filter_sizes))]

        self.dropout: Optional[List[Any]] = None
        if dropout:
            self.dropout = [Dropout(dropout) for i in range(len(filter_sizes))]

    def call(self, input):

        x = input
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.bn:
                x = self.bn[i](x)
            if self.dropout:
                x = self.dropout[i](x)
            x = self.activation(x)
        return x


class ChongTayConvLstmBottleneckBlock(keras.layers.Layer):
    """
    Bottleneck block from https://arxiv.org/abs/1701.01546 - Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
    """

    def __init__(self, filter_sizes: Sequence = (3, 3, 3), n_filters: Sequence = (64, 32, 64),
                 strides: Sequence = (1, 1, 1), padding: str = 'same', dropout: float = 0.,
                 recurrent_dropout: float = 0., activation: Callable = tf.nn.relu, do_permute: bool = True):
        """

        Parameters
        ----------
        filter_sizes: sequence of ConvLSTM filter sizes
        n_filters: sequence of filter numbers for all subsequent layers
        strides: sequence of stride values in x/y dimentions for subsequent ConvLSTM layers
        padding: type of padding, probably should stay as default
        dropout: amount of dropout for Conv part of all ConvLSTM layers
        recurrent_dropout: amount of dropout for LSTM part of all ConvLSTM layers
        activation: tensorflow activation function callable
        do_permute: by default true, ConvLSTM requires different ordering than Conv, this parameter ensures that
         connecting conv output to this bottleneck will work
        """
        if not check_equal_lengths(filter_sizes, n_filters, strides):
            raise Exception("All iterable arguments should be of equal length")
        super(ChongTayConvLstmBottleneckBlock, self).__init__()
        self.layers = [ConvLSTM2D(filters=n, kernel_size=ks, strides=stride, dropout=dropout, padding=padding,
                                  recurrent_dropout=recurrent_dropout, return_sequences=True)
                       for n, ks, stride in zip(n_filters, filter_sizes, strides)]
        self.activation = activation
        self.do_permute = do_permute

    def call(self, input):
        x = input
        if self.do_permute:
            x = Permute((3, 1, 2, 4))(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        if self.do_permute:
            x = Permute((2, 3, 1, 4))(x)
        return x


class ChongTayDecoder(keras.layers.Layer):
    """
    Decoder from https://arxiv.org/abs/1701.01546 - Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
    """

    def __init__(self, filter_sizes: Sequence = (5, 11), n_filters: Sequence = (128, 1), strides: Sequence = (2, 4),
                 dropout: float = 0., add_batchnorm: bool = False, activation: Callable = tf.nn.relu):
        """

        Parameters
        ----------
        filter_sizes: sequence of conv filter sizes
        n_filters: sequence of filter numbers for all subsequent layers
        strides: sequence of stride values in x/y dimentions for subsequent conv layers
        dropout: amount of dropout for all decoder layers
        add_batchnorm: whether to add batchnorm after all decoder layers
        activation: tensorflow activation function callable
        """
        if not check_equal_lengths(filter_sizes, n_filters, strides):
            raise Exception("All iterable arguments should be of equal length")
        super(ChongTayDecoder, self).__init__()
        self.conv_layers = [Conv3DTranspose(filters=n, kernel_size=(ks, ks, 1), strides=(stride, stride, 1))
                            for n, ks, stride in zip(n_filters, filter_sizes, strides)]
        self.activation = activation

        self.bn: Optional[List[Any]] = None
        if add_batchnorm:
            self.bn = [BatchNormalization() for i in range(len(filter_sizes))]

        self.dropout: Optional[List[Any]] = None
        if dropout:
            self.dropout = [Dropout(dropout) for i in range(len(filter_sizes))]

    def call(self, input):

        x = input
        n_layers = len(self.conv_layers)
        for i in range(n_layers):
            x = self.conv_layers[i](x)
            if self.bn:
                x = self.bn[i](x)
            if self.dropout:
                x = self.dropout[i](x)
            if i < n_layers - 1:
                x = self.activation(x)
            else:
                x = tf.nn.sigmoid(x)
        return x
