from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
import tensorflow as tf
import pathlib

from tensorflow.python.data.ops.dataset_ops import ConcatenateDataset


class SinkConfig(NamedTuple):
    path: str


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
        tf.data.experimental.save(
            dataset=dataset.unbatch(), path=str(self.config.path))
