from abc import ABC
from tensorflow._api.v2 import data
from tensorflow.python.data.ops.dataset_ops import BatchDataset

import skvideo.io as sk 
import wandb
import numpy as np
import tensorflow as tf


class Source(ABC):
    def load(self) -> np.ndarray:
        pass


class LocalVideoSource(Source):
    def load(self, path: str) -> np.ndarray:
        vid = sk.vread(path)
        return vid


class WANDBSource(Source):
    def load(self) -> np.ndarray:
        return np.ndarray()


class LocalTFDataSource(Source):
    def load(self, path: str, batch_size: int = None) -> BatchDataset:
        dataset = tf.data.experimental.load(path)
        if batch_size:
            return dataset.batch(batch_size)
        else:
            return dataset