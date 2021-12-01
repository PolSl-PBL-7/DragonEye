from pathlib import Path
import os

from typing import NamedTuple

import tensorflow as tf

from dnn.training.builder import CompileConfig, model_builder
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoderConfig, SpatioTemporalAutoencoder

from data import LocalTFDataSource, SourceConfig

import wandb
from wandb.keras import WandbCallback

from utils.logging_utils import initialize_logger


def training_pipeline(pipeline_params: dict, compile_params: dict, model_params: dict, source_params: dict, training_params: dict):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    source_config = SourceConfig(**source_params)
    source = LocalTFDataSource(source_config)
    dataset = source(pipeline_params['dataset_path'])

    compile_config = CompileConfig(**compile_params)
    model_config = SpatioTemporalAutoencoderConfig(**model_params)
    model = model_builder[pipeline_params['model']](
        model_config=model_config,
        compile_config=compile_config
    )

    train_dataset = tf.data.Dataset.zip((dataset, dataset))
    history = model.fit(train_dataset, **training_params)

    return model, history


if __name__ == '__main__':

    project_dir = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]

    pipeline_params = {
        'dataset_path': project_dir / r'datasets/tf_datasets/avenue_dataset_training_11-30-2021-21-06-25',
        'model': SpatioTemporalAutoencoder.__class__.__name__
    }

    compile_params = {
        'optimizer_params': {},
        'loss_params': {},
        'loss': 'mse',
        'optimizer': 'adam',
        'metrics': ['mse', 'msle', 'mape']
    }

    model_params = {
        'strides_encoder': (1, 1),
        'strides_decoder': (1, 1)
    }

    source_params = {
        'batch_size': 16,
        'fps': 5
    }

    wandb.init(project="trainings", entity="polsl-pbl-7", magic=True, config={**pipeline_params, **compile_params, **model_params, **source_params})

    training_params = {
        'callbacks': [WandbCallback(monitor="loss")],
        'epochs': 20
    }

    # initialize_logger(output_dir=sink_params['path'], args_dict={
    #     'source_params': source_params,
    #     'pipeline_params': pipeline_params,
    #     'training_params' : training_params,
    #     'model_params' : model_params,
    #     'compile_params' : compile_params
    # }
    # )

    history, model = training_pipeline(
        pipeline_params=pipeline_params,
        compile_params=compile_params,
        model_params=model_params,
        source_params=source_params,
        training_params=training_params
    )
