NAME = "data_processing_pipeline"


def data_processing_pipeline(
    versioner_params: dict,
    source_params: dict,
    processor_params: dict,
    pipeline_params: dict,
    sink_params: dict = None,
    source_params_dynamic: dict = None,
    processor_params_dynamic: dict = None,
):
    from data import VersioningConfig, WandbDatasetVersioner,\
        LocalVideoSource, LocalTFDataSource, SourceConfig,\
        VideoProcessor, ProcessorConfig,\
        LocalTFDatasetSink, SinkConfig,\
        DataProcessing, DataProcessingConfig
    import tensorflow as tf

    db = WandbDatasetVersioner()
    config = VersioningConfig(**versioner_params)
    db.load_dataset(config)

    # dataset preparation
    with tf.device('/CPU:0'):
        if sink_params:
            sink_config = SinkConfig(**sink_params)
            sink_tf = LocalTFDatasetSink(sink_config)
        else:
            sink_config = None
            sink_tf = None
        if source_params_dynamic:
            # create ITAE dataset
            # static part
            print('starting job "create ITAE dataset"')
            print('starting job "create static dataset"')
            static_source_config = SourceConfig(**source_params)
            static_source = LocalVideoSource(static_source_config)

            static_processor_config = ProcessorConfig(**processor_params)
            static_processor = VideoProcessor(static_processor_config)

            static_data_processing_config = DataProcessingConfig(
                source=static_source,
                source_config=static_source_config,
                processor=static_processor,
                processor_config=static_processor_config,
                sink=sink_tf,
                sink_config=sink_config,
                **pipeline_params
            )
            static_data_processing = DataProcessing()
            static_dataset = static_data_processing(config=static_data_processing_config)
            print('finished job "create static dataset"')
            # dynamic part
            print('starting job "create dynamic dataset"')
            dynamic_source_config = SourceConfig(**source_params_dynamic)
            dynamic_source = LocalVideoSource(dynamic_source_config)

            dynamic_processor_config = ProcessorConfig(**processor_params_dynamic)
            dynamic_processor = VideoProcessor(dynamic_processor_config)

            dynamic_data_processing_config = DataProcessingConfig(
                source=dynamic_source,
                source_config=dynamic_source_config,
                processor=dynamic_processor,
                processor_config=dynamic_processor_config,
                sink=sink_tf,
                sink_config=sink_config,
                **pipeline_params
            )
            dynamic_data_processing = DataProcessing()
            dynamic_dataset = dynamic_data_processing(config=dynamic_data_processing_config)
            print('finished job "create dynamic dataset"')

            dataset = tf.data.Dataset.zip((static_dataset, dynamic_dataset)).map(lambda s, d: {"Input_Static": s, "Input_Dynamic": d})
            print('finished job "create ITAE dataset"')
        else:
            # create normal dataset
            print('starting job "create dataset"')
            source_config = SourceConfig(**source_params)
            source = LocalVideoSource(source_config)

            processor_config = ProcessorConfig(**processor_params)
            processor = VideoProcessor(processor_config)

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
            print('finished job "create dataset"')
    return dataset
