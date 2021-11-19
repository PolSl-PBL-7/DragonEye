from tensorflow.python.data.ops.dataset_ops import BatchDataset, ConcatenateDataset

from pathlib import Path
import numpy as np

from data.processing.component import DataProcessing, DataProcessingConfig
from data.processing.source import LocalVideoSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig

CURDIR = Path(__file__).parents[1]
dataset_path = CURDIR / "test_videos"


def test_instance():
    source_config = SourceConfig()
    source = LocalVideoSource(source_config)
    data_processing = DataProcessing()
    processor_config = ProcessorConfig()
    processor = VideoProcessor(processor_config)
    data_processing_config = DataProcessingConfig(source=source, source_config=source_config, input=dataset_path,
                                                  processor=processor, processor_config=processor_config)

    dataset = data_processing(config=data_processing_config)

    assert isinstance(dataset, ConcatenateDataset)


def test_shape():
    source_config = SourceConfig()
    source = LocalVideoSource(source_config)
    data_processing = DataProcessing()
    processor_config = ProcessorConfig()
    processor = VideoProcessor(processor_config)
    data_processing_config = DataProcessingConfig(source=source, source_config=source_config, input=dataset_path,
                                                  processor=processor, processor_config=processor_config)

    dataset = data_processing(config=data_processing_config)
    for batch in dataset.take(1):
        assert batch.shape == (processor_config.batch_size, processor_config.time_window, *processor_config.shape, 3)
