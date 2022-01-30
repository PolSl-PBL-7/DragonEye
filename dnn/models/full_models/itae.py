from typing import Any, Sequence, Callable, NamedTuple, Tuple, Optional

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Concatenate, Conv3DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid

from dnn.models.conv.blocks import ItaeEncoderBlock, ItaeDecoderBlock, ConvConfig
from dnn.models.model import Model


class ITAEConfig(NamedTuple):
    pass


class ITAE(Model):
    def __init__(self, model_config: ITAEConfig, compile_config: Any):
        super(ITAE, self).__init__(model_config, compile_config)
        self.model_type = __class__
        print(self.model_type)
        self.__name__ = 'ITAE'

        dc_1 = ConvConfig(12, (5, 3, 3), (1, 2, 2), relu, "Dynamic_Block_1")
        dc_2 = ConvConfig(16, (3, 3, 3), (1, 2, 2), relu, "Dynamic_Block_2")
        dc_3 = ConvConfig(32, (3, 3, 3), (1, 2, 2), relu, "Dynamic_Block_3")
        dc_4 = ConvConfig(32, (3, 3, 3), (1, 1, 1), relu, "Dynamic_Block_4")

        sc_1 = ConvConfig(96, (1, 3, 3), (1, 2, 2), relu, "Static_Block_1")
        sc_2 = ConvConfig(128, (1, 3, 3), (1, 2, 2), relu, "Static_Block_2")
        sc_3 = ConvConfig(256, (1, 3, 3), (1, 2, 2), relu, "Static_Block_3")
        sc_4 = ConvConfig(256, (1, 3, 3), (1, 1, 1), relu, "Static_Block_4")

        dec_1 = ConvConfig(256, (3, 3, 3), (1, 1, 1), activation=relu, name='Decoder_Conv_Transpose_1')
        dec_2 = ConvConfig(128, (3, 3, 3), (2, 2, 2), activation=relu, name='Decoder_Conv_Transpose_2')
        dec_3 = ConvConfig(96, (3, 3, 3), (2, 2, 2), activation=relu, name='Decoder_Conv_Transpose_3')
        dec_4 = ConvConfig(1, (3, 3, 3), (1, 2, 2), activation=sigmoid, name='Decoder_Conv_Transpose_4')

        self.encoder_block_1 = ItaeEncoderBlock(dc_1, sc_1)
        self.encoder_block_2 = ItaeEncoderBlock(dc_2, sc_2)
        self.encoder_block_3 = ItaeEncoderBlock(dc_3, sc_3)
        self.encoder_block_4 = ItaeEncoderBlock(dc_4, sc_4)

        self.decoder_block_1 = ItaeDecoderBlock(dec_1)
        self.decoder_block_2 = ItaeDecoderBlock(dec_2)
        self.decoder_block_3 = ItaeDecoderBlock(dec_3)
        self.decoder_block_4 = ItaeDecoderBlock(dec_4)

    def call(self, input):

        # static, dynamic = self.static_input(input['Input_Static']), self.dynamic_input(input['Input_Dynamic'])
        static, dynamic = self.encoder_block_1(input['Input_Static'], input['Input_Dynamic'])
        static, dynamic = self.encoder_block_2(static, dynamic)
        static, dynamic = self.encoder_block_3(static, dynamic)
        static, dynamic = self.encoder_block_4(static, dynamic)

        out = self.decoder_block_1(static)
        out = self.decoder_block_2(out)
        out = self.decoder_block_3(out)
        out = self.decoder_block_4(out)

        return out
