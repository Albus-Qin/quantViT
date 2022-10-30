# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer
from .log2_for_norm import Log2QuantizerForNorm

str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer, 'log2_for_norm': Log2QuantizerForNorm}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
