import tensorflow as tf
from pathlib import Path

from data.processing.component import DataProcessing, DataProcessingConfig
from data.processing.source import LocalVideoSource, SourceConfig
from data.processing.process import VideoProcessor, ProcessorConfig
from inference import Predictor, PredictorConfig, AnomalyScoreConfig, AnomalyScoreHeuristic
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, ModelConfig

CURDIR = Path(__file__).parents[0]
dataset_path = CURDIR / "test_videos"


def test_full_experiment():

    # setup machine
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

    # dataset preparation
    source_config = SourceConfig(batch_size=8, fps=5)
    source = LocalVideoSource(source_config)

    data_processing = DataProcessing()

    processor_config = ProcessorConfig(shape=(127, 127), time_window=3, batch_size=16)
    processor = VideoProcessor(processor_config)

    data_processing_config = DataProcessingConfig(source=source, source_config=source_config, input=dataset_path, processor=processor, processor_config=processor_config)
    dataset = data_processing(config=data_processing_config)

    for batch in dataset.take(1):
        assert(batch.shape == (processor_config.batch_size, processor_config.time_window, *processor_config.shape, 3))

    train_dataset = tf.data.Dataset.zip((dataset, dataset))

    # model setup
    model_config = ModelConfig(strides_encoder=(2, 2), strides_decoder=(2, 2))
    model = SpatioTemporalAutoencoder(model_config)
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_dataset, epochs=1)

    # predictor setup
    anomaly_score_config = AnomalyScoreConfig()
    anomaly_score = AnomalyScoreHeuristic(anomaly_score_config)

    predictor_config = PredictorConfig(
        reconstruction_model=model,
        anomaly_score=anomaly_score
    )
    predictor = Predictor(predictor_config)
    scores = predictor(dataset)

    for input, score in tf.data.Dataset.zip((dataset, scores)).take(2):
        assert input.shape, (16, 3, 127, 127, 3)
        assert score.shape, (16, 1)
