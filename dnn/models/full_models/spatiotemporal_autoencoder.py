from typing import Sequence, Callable, NamedTuple, Tuple, Optional

import tensorflow as tf

from dnn.models.conv.blocks import ChongTayEncoder, ChongTayConvLstmBottleneckBlock, ChongTayDecoder

# TODO: change config name, add __init__


class SpatioTemporalAutoencoderConfig(NamedTuple):

    filter_sizes_encoder: Sequence = (11, 5)
    n_filters_encoder: Sequence = (128, 64)
    strides_encoder: Sequence = (4, 2)
    dropout_encoder: float = 0.
    add_batchnorm_encoder: bool = False

    filter_sizes_bottleneck: Sequence = (3, 3, 3)
    n_filters_bottleneck: Sequence = (64, 32, 64)
    strides_bottleneck: Sequence = (1, 1, 1)
    dropout_bottleneck: float = 0.
    recurrent_dropout: float = 0.
    bottleneck_layers: int = 2

    filter_sizes_decoder: Sequence = (5, 11)
    n_filters_decoder: Sequence = (128, 1)
    strides_decoder: Sequence = (2, 4)
    dropout_decoder: float = 0.
    add_batchnorm_decoder: bool = False

    activation: Callable = tf.nn.relu


class SpatioTemporalAutoencoder(tf.keras.Model):
    def __init__(self, config: SpatioTemporalAutoencoderConfig):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(ChongTayEncoder(
            filter_sizes=config.filter_sizes_encoder,
            strides=config.strides_encoder,
            n_filters=config.n_filters_encoder,
            dropout=config.dropout_encoder,
            add_batchnorm=config.add_batchnorm_encoder,
            activation=config.activation
        ))
        for i in range(config.bottleneck_layers):
            self.model.add(ChongTayConvLstmBottleneckBlock(
                filter_sizes=config.filter_sizes_bottleneck,
                strides=config.strides_bottleneck,
                n_filters=config.n_filters_bottleneck,
                dropout=config.dropout_bottleneck,
                recurrent_dropout=config.recurrent_dropout,
                activation=config.activation
            ))
        self.model.add(ChongTayDecoder(
            filter_sizes=config.filter_sizes_decoder,
            strides=config.strides_decoder,
            n_filters=config.n_filters_decoder,
            dropout=config.dropout_decoder,
            add_batchnorm=config.add_batchnorm_decoder,
            activation=config.activation
        ))

    @classmethod
    def create_from_configs(cls, model_config, compile_config):
        from dnn.training.losses import losses
        from dnn.training.metrics import metrics
        from dnn.training.optimizers import optimizers

        model = cls(model_config)
        model.compile(
            loss=losses[compile_config.loss](**compile_config.loss_params),
            optimizer=optimizers[compile_config.optimizer](**compile_config.optimizer_params),
            metrics=[metrics[key] for key in compile_config.metric_list]
        )
        return model

    def call(self, input):
        return self.model(input)

    def __name__(self):
        return 'spatiotemporal_autoencoder'
