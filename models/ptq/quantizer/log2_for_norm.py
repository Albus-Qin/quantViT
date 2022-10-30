import torch

from .base import BaseQuantizer


class Log2QuantizerForNorm(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(Log2QuantizerForNorm, self).__init__(
            bit_type,
            observer,
            module_type,
        )
        self.max_abs = None
        self.sign = None
        self.softmax_mask = None

    def quant(self, x):
        self.sign = x.sign()
        self.max_abs = x.abs().max(dim=-1,keepdim=True).values
        x = x.abs() / self.max_abs
        x = torch.round(-1 * x.log2())
        self.softmax_mask = x >= 2**(self.bit_type.bits-1)
        x = self.sign*torch.clamp(x, 0, 2**(self.bit_type.bits-1) - 1)
        return x

    def dequantize(self, x):
        x = x.abs()
        x = 2**(-1 * x) * self.max_abs
        x = self.sign * x
        x[self.softmax_mask] = 0
        return x
