from pathlib import Path
import os

from DragonEye.inference import anomaly_score, predictor
from DragonEye.inference.anomaly_score import AnomalyScore
from inference import Predictor, PredictorConfig, AnomalyScoreHeuristic, AnomalyScoreConfig
from data import LocalTFDataSource, SourceConfig, LocalTFDatasetSink, SinkConfig

import tensorflow as tf


def prediction_pipeline(source_params, anomaly_score_params, sink_params, pipeline_params):

    # get reconstruction model
    model = tf.keras.models.load_model(pipeline_params['model_path'])

    # get dataset
    source_config = SourceConfig(**source_params)
    source = LocalTFDataSource(source_config)
    dataset = source(pipeline_params['dataset_path'])

    # prepare predictor
    anomaly_score_config = AnomalyScoreConfig(**anomaly_score_params)
    anomaly_score = AnomalyScoreHeuristic(anomaly_score_config)
    predictor_config = PredictorConfig(
        reconstruction_model=model,
        anomaly_score=anomaly_score
    )
    predictor = Predictor(predictor_config)

    # get anomaly scores
    scores = predictor(dataset=dataset)

    # save predictions as tf dataset file
    sink_config = SinkConfig(**sink_params)
    sink = LocalTFDatasetSink(sink_config)
    sink(scores)
    return scores


if __name__ == "__main__":
    from datetime import datetime
    from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder

    main_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]

    source_params = {
        'batch_size': 16,
        'fps': 5
    }

    anomaly_score_params = {}

    sink_params = sink_params = {
        'path': main_path / 'experiments' / 'predictions' / f'avenue_dataset_train_{datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S")}'
    }

    pipeline_params = {
        'model_path': main_path / 'experiments' / 'models' / SpatioTemporalAutoencoder.__name__ / '12-01-2021-14-32-50',
        'dataset_path': main_path / 'experiments' / 'datasets' / 'tf_datasets' / 'avenue_dataset_training_12-01-2021-11-51-02'
    }

    prediction_pipeline(
        source_params=sink_params,
        anomaly_score_params=anomaly_score_params,
        sink_params=sink_params,
        pipeline_params=pipeline_params
    )
