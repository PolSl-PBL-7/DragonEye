import tensorflow as tf
from typing import NamedTuple, Union, List, Optional

from dnn.training.losses import losses
from dnn.training.metrics import metrics
from dnn.training.optimizers import optimizers

from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, SpatioTemporalAutoencoderConfig

class CompileConfig(NamedTuple):
    optimizer_params: dict = {}
    loss_params: dict = {}
    loss: str = 'mse'
    optimizer: str = 'adam'
    metrics: List[str] = ['mse', 'msle', 'mape']

model_builder = {
    SpatioTemporalAutoencoder.__class__.__name__: SpatioTemporalAutoencoder.create_from_configs
}