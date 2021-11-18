from tensorflow.python.data.ops.dataset_ops import BatchDataset

from pathlib import Path
import numpy as np

from data.processing.source import LocalVideoSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig

CURDIR = Path(__file__).parents[1]
dataset_path = CURDIR/"test_videos"

def test_instance():
    source_config = SourceConfig()
    source = LocalVideoSource(config=source_config)
    processor_config = ProcessorConfig()
    processor = VideoProcessor(processor_config)

    video = source(path = dataset_path/'15.avi')
    processed_video = processor(video)

    assert(isinstance(processed_video, BatchDataset), True) 

def test_shape():
    source_config = SourceConfig()
    source = LocalVideoSource(config=source_config)
    processor_config = ProcessorConfig()
    processor = VideoProcessor(processor_config)

    video = source(path = dataset_path/'15.avi')
    for batch in processor(video).take(1):
        assert(batch.shape == (processor_config.batch_size, processor_config.time_window, *processor_config.shape, 3))
