NAME = "data_processing_pipeline"


def data_processing_pipeline(
    versioner_params: dict,
    source_params: dict,
    processor_params: dict,
    sink_params: dict,
    pipeline_params: dict
):
    from data import VersioningConfig, WandbDatasetVersioner,\
        LocalVideoSource, LocalTFDataSource, SourceConfig,\
        VideoProcessor, ProcessorConfig,\
        LocalTFDatasetSink, SinkConfig,\
        DataProcessing, DataProcessingConfig

    db = WandbDatasetVersioner()
    config = VersioningConfig(**versioner_params)
    db.load_dataset(config)

    # dataset preparation
    source_config = SourceConfig(**source_params)
    source = LocalVideoSource(source_config)

    processor_config = ProcessorConfig(**processor_params)
    processor = VideoProcessor(processor_config)

    sink_config = SinkConfig(**sink_params)
    sink_tf = LocalTFDatasetSink(sink_config)

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
