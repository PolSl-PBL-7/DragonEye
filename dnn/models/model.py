import tensorflow as tf
import warnings

from tensorflow.python.eager import context
from keras import backend
from keras import callbacks as callbacks_module
from keras.engine import base_layer
from keras.engine import base_layer_utils
from keras.engine import data_adapter
from keras.utils import traceback_utils
from keras.utils import version_utils

from utils.file_loaders import Pickle


def _disallow_inside_tf_function(method_name):
    if tf.inside_function():
        error_msg = (
            'Detected a call to `Model.{method_name}` inside a `tf.function`. '
            '`Model.{method_name} is a high-level endpoint that manages its own '
            '`tf.function`. Please move the call to `Model.{method_name}` outside '
            'of all enclosing `tf.function`s. Note that you can call a `Model` '
            'directly on `Tensor`s inside a `tf.function` like: `model(x)`.'
        ).format(method_name=method_name)
        raise RuntimeError(error_msg)


def _is_tpu_multi_host(strategy):
    return (backend.is_tpu_strategy(strategy) and strategy.extended.num_hosts > 1)


def concat(tensors, axis=0):
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], tf.SparseTensor):
        return tf.sparse.concat(axis=axis, sp_inputs=tensors)
    return tf.concat(tensors, axis=axis)


class Model(tf.keras.Model):
    model_config = None
    compile_config = None
    model_type = None

    def __init__(self, model_config, compile_config):
        super(Model, self).__init__()
        self.compile_config = compile_config
        self.model_config = model_config
        self.model_type = __class__

    def call(self, inputs):
        return inputs

    @classmethod
    def create_from_configs(cls, model_config, compile_config):
        from dnn.training.losses import losses
        from dnn.training.metrics import metrics
        from dnn.training.optimizers import optimizers

        model = cls(model_config, compile_config)
        model.compile(
            loss=losses[compile_config.loss](**compile_config.loss_params),
            optimizer=optimizers[compile_config.optimizer](**compile_config.optimizer_params),
            metrics=[metrics[key] for key in compile_config.metric_list]
        )
        return model

    @traceback_utils.filter_traceback
    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """Generates output predictions for the input samples.
        Computation is done in batches. This method is designed for performance in
        large scale inputs. For small amount of inputs that fit in one batch,
        directly using `__call__()` is recommended for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `tf.keras.layers.BatchNormalization` that behaves differently during
        inference. Also, note the fact that test loss is not affected by
        regularization layers like noise and dropout.
        Args:
            x: Input samples. It could be:
            - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
            - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
            - A `tf.data` dataset.
            - A generator or `keras.utils.Sequence` instance.
            A more detailed description of unpacking behavior for iterator types
            (Dataset, generator, Sequence) is given in the `Unpacking behavior
            for iterator-like inputs` section of `Model.fit`.
            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`. If x is a `tf.data`
                dataset and `steps` is None, `predict()` will
                run until the input dataset is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        See the discussion of `Unpacking behavior for iterator-like inputs` for
        `Model.fit`. Note that Model.predict uses the same interpretation rules as
        `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for all
        three methods.
        Returns:
            Numpy array(s) of predictions.
        Raises:
            RuntimeError: If `model.predict` is wrapped in a `tf.function`.
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.
        """
        base_layer.keras_api_gauge.get_cell('predict').set(True)
        version_utils.disallow_legacy_graph('Model', 'predict')
        self._check_call_args('predict')
        _disallow_inside_tf_function('predict')

        # TODO(yashkatariya): Cache model on the coordinator for faster prediction.
        # If running under PSS, then swap it with OneDeviceStrategy so that
        # execution will run on the coordinator.
        original_pss_strategy = None
        if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
            original_pss_strategy = self.distribute_strategy
            self._distribution_strategy = None

        # Cluster coordinator is set by `.fit()` and `.evaluate()` which is not
        # needed in `.predict()` because all the predictions happen on the
        # coordinator/locally.
        if self._cluster_coordinator:
            self._cluster_coordinator = None

        outputs = None
        with self.distribute_strategy.scope():
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            dataset_types = (tf.compat.v1.data.Dataset, tf.data.Dataset)
            if (self._in_multi_worker_mode() or _is_tpu_multi_host(
                    self.distribute_strategy)) and isinstance(x, dataset_types):
                try:
                    options = tf.data.Options()
                    data_option = tf.data.experimental.AutoShardPolicy.DATA
                    options.experimental_distribute.auto_shard_policy = data_option
                    x = x.with_options(options)
                except ValueError:
                    warnings.warn(
                        'Using Model.predict with '
                        'MultiWorkerDistributionStrategy or TPUStrategy and '
                        'AutoShardPolicy.FILE might lead to out-of-order result'
                        '. Consider setting it to AutoShardPolicy.DATA.',
                        stacklevel=2)

            data_handler = data_adapter.get_data_handler(
                x=x,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=1,
                    steps=data_handler.inferred_steps)

            self.predict_function = self.make_predict_function()
            self._predict_counter.assign(0)
            callbacks.on_predict_begin()
            batch_outputs = None
            for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        callbacks.on_predict_batch_begin(step)
                        tmp_batch_outputs = self.predict_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
                        if outputs is None:
                            outputs = tf.data.Dataset.from_tensor_slices(tf.expand_dims(batch_outputs, 0))
                        else:
                            tf.data.Dataset.concatenate([
                                tf.data.Dataset.from_tensor_slices(tf.expand_dims(batch_outputs, 0)),
                                outputs])
                        end_step = step + data_handler.step_increment
                        callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
            if batch_outputs is None:
                raise ValueError('Unexpected result of `predict_function` '
                                 '(Empty batch_outputs). Please use '
                                 '`Model.compile(..., run_eagerly=True)`, or '
                                 '`tf.config.run_functions_eagerly(True)` for more '
                                 'information of where went wrong, or file a '
                                 'issue/bug to `tf.keras`.')
            callbacks.on_predict_end()

        # If originally PSS strategy was used, then replace it back since predict
        # is running under `OneDeviceStrategy` after the swap and once its done
        # we need to replace it back to PSS again.
        if original_pss_strategy is not None:
            self._distribution_strategy = original_pss_strategy

        return outputs

    def save(self, path):
        self.save_weights(f'{path}/weights')
        Pickle.save(self.model_config, f'{path}/model_config')
        Pickle.save(self.compile_config, f'{path}/compile_config')
        Pickle.save(self.model_type, f'{path}/model_type')


def load_model(path):
    from dnn.training.losses import losses
    from dnn.training.metrics import metrics
    from dnn.training.optimizers import optimizers
    model_config = Pickle.load(f'{path}/model_config')
    compile_config = Pickle.load(f'{path}/compile_config')
    model_type = Pickle.load(f'{path}/model_type')
    model = model_type(model_config, compile_config)
    print(type(model))
    model.compile(
        loss=losses[compile_config.loss](**compile_config.loss_params),
        optimizer=optimizers[compile_config.optimizer](**compile_config.optimizer_params),
        metrics=[metrics[key] for key in compile_config.metric_list]
    )
    model.load_weights(f'{path}/weights')
    return model
