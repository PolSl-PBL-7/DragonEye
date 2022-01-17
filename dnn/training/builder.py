import tensorflow as tf
from typing import NamedTuple, Union, List, Optional

from dnn.training.losses import losses
from dnn.training.metrics import metrics
from dnn.training.optimizers import optimizers

from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, SpatioTemporalAutoencoderConfig
from dnn.models.full_models.itae import ITAE, ITAEConfig


class CompileConfig(NamedTuple):
    optimizer_params: dict = {}
    loss_params: dict = {}
    loss: str = 'mse'
    optimizer: str = 'adam'
    metric_list: List[str] = ['mae', 'msle', 'mape']


model_builder = {
    SpatioTemporalAutoencoder.__name__: SpatioTemporalAutoencoder.create_from_configs,
    ITAE.__name__: ITAE.create_from_configs
}

config_builder = {
    SpatioTemporalAutoencoder.__name__: SpatioTemporalAutoencoderConfig,
    ITAE.__name__: ITAEConfig
}