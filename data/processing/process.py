from typing import Tuple, NamedTuple
from tensorflow.python.data.ops.dataset_ops import BatchDataset

import numpy as np
import tensorflow as tf


class ProcessorConfig(NamedTuple):
    shape: Tuple[int, int] = (227, 227)
    time_window: int = 10
    batch_size: int = 32


class VideoProcessor:
    shape: Tuple[int, int]
    time_window: int
    batch_size: int
    padding: str

    def __init__(self, config: ProcessorConfig):
        """
        Object used to adjust shape, create time windows and batch data.
        In the future it may be extended with different masks/filters.

        Args:
            shape (tuple, optional): Width and height that video should be adjusted to. Defaults to (320, 320).
            time_window (int, optional): Nuber of frames per example. Defaults to 10.
            batch_size (int, optional): Number of examples/time windows per batch of data. Defaults to 128.
        """
        self.shape=config.shape
        self.time_window=config.time_window
        self.batch_size=config.batch_size

    def __call__(self, vid: np.ndarray) -> BatchDataset:
        """Method for applying preprocessing per recording.

        Args:
            vid (np.ndarray): Video in the form of numpy array, of shape (frames, height, width, channels)

        Returns:
            BatchDataset: Tensorflow Dataset
        """
        vid_resized = tf.image.resize(vid, self.shape)
        vid_resized /= 255.
        vid_windowed_and_batched = tf.keras.utils.timeseries_dataset_from_array(vid_resized, None, self.time_window, 1, 1, self.batch_size)
        return vid_windowed_and_batched
