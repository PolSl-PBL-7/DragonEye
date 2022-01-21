NAME = "training_pipeline"


def training_pipeline(
    pipeline_params: dict,
    compile_params: dict,
    model_params: dict,
    source_params: dict,
    training_params: dict,
    data_processing_pipeline_params:
    dict = None,
    versioner_params: dict = None,
    processor_params: dict = None,
    sink_params: dict = None
):

    from dnn.training.builder import CompileConfig, model_builder
    from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoderConfig, SpatioTemporalAutoencoder
    from data import LocalTFDataSource, SourceConfig
    from pipelines.data_processing_pipeline import data_processing_pipeline

    import tensorflow as tf

    import wandb
    from wandb.keras import WandbCallback

    from utils.callbacks import CallbackName, get_callback_by_name
    from datetime import datetime
    import pickle

    with tf.device('/CPU:0'):
        if data_processing_pipeline_params and versioner_params and processor_params:
            dataset = data_processing_pipeline(
                versioner_params=versioner_params,
                source_params=source_params,
                processor_params=processor_params,
                pipeline_params=data_processing_pipeline_params,
                sink_params=sink_params)
        else:
            source_config = SourceConfig(**source_params)
            source = LocalTFDataSource(source_config)
            dataset = source(pipeline_params['dataset_path'])

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(
                    logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dataset = tf.data.Dataset.zip((dataset, dataset))
    dataset_size = len([0 for _ in dataset])
    train_dataset = dataset.take(int(dataset_size * 0.8))
    val_dataset = dataset.skip(int(dataset_size * 0.8))

    compile_config = CompileConfig(**compile_params)
    model_config = SpatioTemporalAutoencoderConfig(**model_params)
    model = model_builder[pipeline_params['model']](
        model_config=model_config,
        compile_config=compile_config
    )

    for callback in training_params['callbacks']:
        if callback == CallbackName.wandb_training_loss.value:
            wandb.init(project=pipeline_params['project'],
                       entity=pipeline_params['entity'],
                       magic=pipeline_params['magic'])

    training_params['callbacks'] = [callback if not isinstance(callback, str) else get_callback_by_name(callback) for callback in training_params['callbacks']]

    history = model.fit(train_dataset, **training_params, validation_data=val_dataset, shuffle=True)

    model_path = str(pipeline_params['model_path'])
    if pipeline_params['add_date_to_model_path']:
        model_path += f'/{datetime.now().strftime(r"%m-%d-%Y-%H-%M-%S")}'

    if pipeline_params['model_path']:
        model.save(model_path + '/model')
        with open(model_path + '/history', 'wb') as f:
            pickle.dump(history, f)

    return model, history
