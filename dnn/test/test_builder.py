from dnn.training.builder import CompileConfig, model_builder
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoderConfig


def test_build_spatiotemporal_autoencoder():
    model_config = SpatioTemporalAutoencoderConfig()
    compile_config = CompileConfig()
    model = model_builder['spatiotemporal_autoencoder'](model_config=model_config, compile_config=compile_config)
