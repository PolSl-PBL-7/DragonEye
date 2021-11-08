from typing import Tuple
from tensorflow.python.data.ops.dataset_ops import BatchDataset

import numpy as np
import tensorflow as tf


class VideoProcessor:
    shape: Tuple[int, int]
    time_window: int
    batch_size: int
    padding: str

    def __init__(self, shape = (320, 320), time_window = 10, batch_size = 128, padding = 'zero'):
        """
        Object used to adjust shape, create time windows and batch data. 
        In the future it may be extended with different masks/filters.

        Args:
            shape (tuple, optional): Width and height that video should be adjusted to. Defaults to (320, 320).
            time_window (int, optional): Nuber of frames per example. Defaults to 10.
            batch_size (int, optional): Number of examples/time windows per batch of data. Defaults to 128.
            padding (str, optional): Version of padding that should be used on time dimension. Options: 'zero'; 'none'. Defaults to 'zero'.
        """
        self.shape = shape
        self.time_window = time_window
        self.batch_size = batch_size
        self.padding = padding
    
    def process(self, vid: np.ndarray) -> BatchDataset:
        """Method for applying preprocessing per recording.

        Args:
            vid (np.ndarray): Video in the form of numpy array, of shape (frames, height, width, channels)

        Returns:
            BatchDataset: Tensorflow Dataset
        """
        vid_resized = tf.image.resize(vid, self.shape)
        vid_windowed_and_batched = tf.keras.utils.timeseries_dataset_from_array(vid_resized, None, self.time_window, 1, 1, self.batch_size)
        return vid_windowed_and_batched




