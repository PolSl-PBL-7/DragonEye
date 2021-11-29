import tensorflow as tf
from pathlib import Path

from data.processing.component import DataProcessing, DataProcessingConfig
from data.processing.source import LocalVideoSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig

from inference.predictor import Predictor, PredictorConfig, AnomalyScoreConfig, AnomalyScoreHeuristic
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, ModelConfig

CURDIR = Path(__file__).parents[1]
dataset_path = CURDIR/"test_data"

def test_predictor():

    # dataset preparation
    source_config = SourceConfig()
    source = LocalVideoSource(source_config)

    data_processing = DataProcessing()
    processor_config = ProcessorConfig()

    processor = VideoProcessor(processor_config)
    data_processing_config = DataProcessingConfig(source=source, source_config = source_config, input=dataset_path, processor=processor,processor_config=processor_config)
    dataset = data_processing(config = data_processing_config)

    # model setup 
    model_config = ModelConfig() # TODO: model config params should be based on processing config
    model = SpatioTemporalAutoencoder(model_config)
    model.compile(loss = 'mse', optimizer='adam')
    model.fit(dataset, epochs = 1)

    # predictor setup
    anomaly_score_config = AnomalyScoreConfig()
    anomaly_score = AnomalyScoreHeuristic(anomaly_score_config)

    predictor_config = PredictorConfig(
        reconstruction_model=model,
        anomaly_score=anomaly_score
        )
    predictor = Predictor(predictor_config)

    scores = predictor(dataset)






