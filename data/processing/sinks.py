from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
import pathlib
from datetime import datetime

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import ConcatenateDataset


class SinkConfig(NamedTuple):
    path: str
    add_date_to_path = True


class Sink(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, config: SinkConfig):
        raise NotImplementedError


class LocalTFDatasetSink(Sink):
    def __init__(self, config: SinkConfig):
        """
        Sink object that is used to return processed dataset locally as tfrecord
        """
        self.config = config
        pass

    def __call__(self, dataset: ConcatenateDataset):
        path = str(self.config.path) + f'/{datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S")}' if self.config.add_date_to_path else  str(self.config.path)
        tf.data.experimental.save(
            dataset=dataset.unbatch(), path=path)
