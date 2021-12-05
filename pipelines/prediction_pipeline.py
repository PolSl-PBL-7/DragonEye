NAME = "prediction_pipeline"


def prediction_pipeline(source_params, anomaly_score_params, sink_params, pipeline_params):
    from inference import anomaly_score, predictor
    from inference.anomaly_score import AnomalyScore
    from inference import Predictor, PredictorConfig, AnomalyScoreHeuristic, AnomalyScoreConfig
    from data import LocalTFDataSource, SourceConfig, LocalTFDatasetSink, SinkConfig

    import tensorflow as tf

    # get reconstruction model
    model = tf.keras.models.load_model(str(pipeline_params['model_path'] / 'model'))

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
