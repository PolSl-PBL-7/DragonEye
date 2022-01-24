class Model():

    def __init__(self):
        raise NotImplementedError("This class can not be instantiated")

    def call(self):
        raise NotImplementedError("This class can not be called")

    @classmethod
    def create_from_configs(cls, model_config, compile_config):
        from dnn.training.losses import losses
        from dnn.training.metrics import metrics
        from dnn.training.optimizers import optimizers

        model = cls(model_config)
        model.compile(
            loss=losses[compile_config.loss](**compile_config.loss_params),
            optimizer=optimizers[compile_config.optimizer](**compile_config.optimizer_params),
            metrics=[metrics[key] for key in compile_config.metric_list]
        )
        return model
