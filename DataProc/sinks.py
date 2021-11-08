from abc import ABC

import tensorflow as tf
import pathlib
class Sink(ABC):
    def __init__(self):
        pass

    def sink(self):
        pass

class LocalTFRecordSink(Sink):
    def __init__(self):
        """
        Sink object that is used to return processed dataset locally as tfrecord
        """
        pass

    def sink(self, dataset: tf.data.Dataset, path: str):
        tf.data.experimental.save(dataset.unbatch(), path)


