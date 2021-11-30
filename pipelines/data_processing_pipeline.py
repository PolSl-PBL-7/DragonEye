from data.versioning.versioner import VersioningConfig, WandbDatasetVersioner

from data.processing.component import DataProcessing, DataProcessingConfig
from data.processing.source import LocalVideoSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig

from data.processing.sinks import SinkConfig, LocalTFDatasetSink

from inference import Predictor, PredictorConfig, AnomalyScoreConfig, AnomalyScoreHeuristic
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, ModelConfig
from pathlib import Path

import tensorflow as tf
from datetime import datetime

import os


def data_processing_pipeline(dataset_path: Path, dataset_name: str):

    db = WandbDatasetVersioner()
    config = VersioningConfig(type='folder', dataset_path=dataset_path, dataset_name=dataset_name)
    db.load_dataset(config)

    # dataset preparation
    source_config = SourceConfig(batch_size=8, fps = 5)
    source = LocalVideoSource(source_config)

    data_processing = DataProcessing()

    processor_config = ProcessorConfig(shape=(127, 127), time_window=3, batch_size=16)
    processor = VideoProcessor(processor_config)


    sink_config = SinkConfig(dataset_path / 'tf_dataset' / datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S"))

    sink_tf = LocalTFDatasetSink()

    data_processing_config = DataProcessingConfig(source=source, source_config=source_config, processor=processor, processor_config=processor_config, sink = sink_tf, sink_config=sink_config)
    datasets = data_processing(config = data_processing_config)

    sink_tf(datasets, sink_config)
    return sink_config


if __name__ == "__main__":
    config = data_processing_pipeline(Path(os.path.dirname(os.path.realpath(__file__))).parents[0] / 'datasets', 'avenue-dataset')
    print(config)