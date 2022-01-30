NAME = "prediction_pipeline"


def prediction_pipeline(
    source_params,
    anomaly_score_params,
    pipeline_params,
    sink_params=None,
    data_processing_pipeline_params=None,
    versioner_params=None,
    processor_params=None,
    source_params_dynamic=None,
    processor_params_dynamic=None,
):
    from inference import anomaly_score, predictor
    from inference.anomaly_score import AnomalyScore
    from inference import Predictor, PredictorConfig, AnomalyScoreHeuristic, AnomalyScoreConfig
    from data import LocalTFDataSource, SourceConfig, LocalTFDatasetSink, SinkConfig
    from pipelines.data_processing_pipeline import data_processing_pipeline
    from dnn.models.model import load_model
    import tensorflow as tf

    # get reconstruction model
    model = load_model(str(f"{pipeline_params['model_path']}\model"))

    # get dataset
    if data_processing_pipeline_params and versioner_params and processor_params:
        dataset = data_processing_pipeline(
            versioner_params=versioner_params,
            source_params=source_params,
            processor_params=processor_params,
            pipeline_params=data_processing_pipeline_params,
            sink_params=None,
            source_params_dynamic=source_params_dynamic,
            processor_params_dynamic=processor_params_dynamic)
    else:
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
    dataset, predictions, scores = predictor(dataset=dataset)

    # save predictions as tf dataset file
    if sink_params:
        sink_config = SinkConfig(**sink_params)
        sink = LocalTFDatasetSink(sink_config)
        sink(scores)

    return dataset, predictions, scores
