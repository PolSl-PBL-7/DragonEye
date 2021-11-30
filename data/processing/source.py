from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Sequence, Union
import ast

from tensorflow._api.v2 import data
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from pathlib import Path

import skvideo.io as sk
import wandb
import numpy as np
import tensorflow as tf


class SourceConfig(NamedTuple):
    batch_size: Optional[int] = None
    fps: int = None


class Source(ABC):
    """Abstract class, that represents all data sources"""
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, path: Union[str, Path], config: SourceConfig):
        raise NotImplementedError


class LocalVideoSource(Source):
    """Source that loads single local video"""

    def __init__(self, config: SourceConfig) -> None:
        super().__init__()
        self.config = config

    def __call__(self, path: Union[str, Path]) -> np.ndarray:
        vid = sk.vread(str(path))
        if self.config.fps:
            video_fps = sk.ffprobe(str(path))['video']['@avg_frame_rate']
            frames, seconds = [int(x) for x in video_fps.split("/")]
            video_fps = frames / seconds
            if self.config.fps < video_fps:
                vid = vid[::int(video_fps / self.config.fps)]
        return vid


class LocalTFDataSource(Source):
    """Source that loads single tf dataset"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, path: Union[str, Path], config: SourceConfig) -> BatchDataset:
        dataset = tf.data.experimental.load(str(path))
        if config.batch_size:
            return dataset.batch(config.batch_size)
        else:
            return dataset
