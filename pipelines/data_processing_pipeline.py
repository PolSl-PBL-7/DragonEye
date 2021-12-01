from data import VersioningConfig, WandbDatasetVersioner,\
    LocalVideoSource, LocalTFDataSource, SourceConfig,\
    VideoProcessor, ProcessorConfig,\
    LocalTFDatasetSink, SinkConfig,\
    DataProcessing, DataProcessingConfig

from pathlib import Path
import os

import tensorflow as tf

from utils.logging_utils import initialize_logger


def data_processing_pipeline(
    versioner_params: dict,
    source_params: dict,
    processor_params: dict,
    sink_params: dict,
    pipeline_params: dict
):

    db = WandbDatasetVersioner()
    config = VersioningConfig(**versioner_params)
    db.load_dataset(config)

    # dataset preparation
    source_config = SourceConfig(**source_params)
    source = LocalVideoSource(source_config)

    processor_config = ProcessorConfig(**processor_params)
    processor = VideoProcessor(processor_config)

    sink_config = SinkConfig(**sink_params)
    sink_tf = LocalTFDatasetSink()

    data_processing_config = DataProcessingConfig(
        source=source,
        source_config=source_config,
        processor=processor,
        processor_config=processor_config,
        sink=sink_tf,
        sink_config=sink_config,
        **pipeline_params
    )
    data_processing = DataProcessing()
    dataset = data_processing(config=data_processing_config)

    return dataset


if __name__ == "__main__":
    from datetime import datetime

    main_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

    sink_params = {
        'path': main_path / 'experiments' / 'datasets' / 'tf_datasets' / f'avenue_dataset_training_{datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S")}'
    }

    versioner_params = {
        'project_name': 'avenue-experiments',
        'entity': 'polsl-pbl-7',
        'job_type': 'test',
        'dataset_name': 'avenue-dataset',
        'dataset_path': main_path / 'experiments' / 'datasets' / 'avenue-dataset',
        'type': 'folder',
        'tag': 'latest',
        'artifact_type': 'dataset',
        'experiment_name': 'test'
    }

    source_params = {
        'batch_size': 32,
        'fps': 5
    }

    processor_params = {
        'shape': (64, 64),
        'time_window': 5,
        'batch_size': 32
    }

    pipeline_params = {
        'input': main_path / 'experiments' / 'datasets' / 'avenue-dataset' / 'training_videos',
        'video_extentions': ['mp4', 'avi', 'mov']
    }

    initialize_logger(output_dir=sink_params['path'], args_dict={
        'source_params': source_params,
        'versioner_params': versioner_params,
        'processor_params': processor_params,
        'pipeline_params': pipeline_params,
        'sink_params': sink_params
    }
    )

    data_processing_pipeline(
        versioner_params=versioner_params,
        source_params=source_params,
        processor_params=processor_params,
        sink_params=sink_params,
        pipeline_params=pipeline_params
    )
