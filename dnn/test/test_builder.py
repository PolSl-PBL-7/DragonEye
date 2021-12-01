from dnn.training.builder import CompileConfig, model_builder
from dnn.models.full_models.spatiotemporal_autoencoder import SpatioTemporalAutoencoder, SpatioTemporalAutoencoderConfig


def test_build_spatiotemporal_autoencoder():
    model_config = SpatioTemporalAutoencoderConfig()
    compile_config = CompileConfig()
    model_builder[SpatioTemporalAutoencoder.__class__.__name__](model_config=model_config, compile_config=compile_config)
