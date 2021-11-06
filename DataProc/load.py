import mypy

import skvideo.io as sk 
import wandb

import numpy as np


class VideoLoader(object):
    filepath: str

    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load(self) -> np.array:
        pass

class LocalLoader(VideoLoader):
    def load(self) -> np.ndarray:
        return sk.vread(self.filepath)


class WANDBLoader(VideoLoader):
    def load(self) -> np.array:
        pass
