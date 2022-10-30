# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer


class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QPreSoftmaxAct(nn.Module):

    def __init__(self,
                 num_heads = 8,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QPreSoftmaxAct, self).__init__()
        self.num_heads = num_heads
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.headwise_observer = [build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode) for i in range(num_heads)]
        self.headwise_quantizer = [build_quantizer(self.quantizer_str, self.bit_type,
                                         self.headwise_observer[i], self.module_type) for i in range(num_heads)]
        self.scales = []

    def forward(self, x):
        if self.calibrate:
            self.scales.clear()
            for head in range(self.num_heads):
                head_wise_x = x[:, head, :, :]
                self.headwise_quantizer[head].observer.update(head_wise_x)
                if self.last_calibrate:
                    self.headwise_quantizer[head].update_quantization_params(head_wise_x)
                self.scales.append(self.headwise_quantizer[head].scale)
        if not self.quant:
            return x
        # quantize
        quantized_head_wise_x = []
        for head in range(self.num_heads):
            head_wise_x = x[:, head, :, :]
            quantized_head_wise_x.append(self.headwise_quantizer[head](head_wise_x))
        x = torch.stack(quantized_head_wise_x, dim=1)
        return x



class QPostSoftmaxAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QPostSoftmaxAct, self).__init__()
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observers = [build_observer(self.observer_str, self.module_type,
                                         self.bit_type, self.calibration_mode) for i in range(4)]
        self.quantizers = [build_quantizer(self.quantizer_str, self.bit_type,
                                           self.observers[i], self.module_type) for i in range(4)]

    def forward(self, x):
        softmax_out = x
        mask1 = softmax_out <= 2 ** (-8)
        mask2 = (softmax_out > 2 ** (-8)) & (softmax_out <= 2 ** (-4))
        mask3 = (softmax_out > 2 ** (-4)) & (softmax_out <= 2 ** (-2))
        mask4 = softmax_out > 2 ** (-2)

        x1 = softmax_out.clone()
        x1[~mask1] = 0
        x2 = softmax_out.clone()
        x2[~mask2] = 2 ** (-8)
        x3 = softmax_out.clone()
        x3[~mask3] = 2 ** (-4)
        x4 = softmax_out.clone()
        x4[~mask4] = 2 ** (-2)

        if self.calibrate:
            self.quantizers[0].observer.update(x1)
            self.quantizers[1].observer.update(x2)
            self.quantizers[2].observer.update(x3)
            self.quantizers[3].observer.update(x4)

            if self.last_calibrate:
                self.quantizers[0].update_quantization_params(x1)
                self.quantizers[1].update_quantization_params(x2)
                self.quantizers[2].update_quantization_params(x3)
                self.quantizers[3].update_quantization_params(x4)

        if not self.quant:
            return x
        # quantize
        x1 = self.quantizers[0](x1)
        x2 = self.quantizers[1](x2)
        x3 = self.quantizers[2](x3)
        x4 = self.quantizers[3](x4)

        x1[~mask1] = 0
        x2[~mask2] = 0
        x3[~mask3] = 0
        x4[~mask4] = 0
        return x1 + x2 + x3 + x4


class QPostGELUAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QPostGELUAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer_a = build_observer(self.observer_str, self.module_type,
                                         self.bit_type, self.calibration_mode)
        self.quantizer_a = build_quantizer(self.quantizer_str, self.bit_type,
                                           self.observer_a, self.module_type)
        self.observer_b = build_observer(self.observer_str, self.module_type,
                                         self.bit_type, self.calibration_mode)
        self.quantizer_b = build_quantizer(self.quantizer_str, self.bit_type,
                                           self.observer_b, self.module_type)

    def forward(self, x):
        if self.calibrate:
            a = x.clone()
            a[a >= 0] = 0
            b = x.clone()
            b[b < 0] = 0
            self.quantizer_a.observer.update(a)
            self.quantizer_b.observer.update(b)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer_a.update_quantization_params(a)
                self.quantizer_b.update_quantization_params(b)
        if not self.quant:
            return x
        a = x.clone()
        a[a >= 0] = 0
        b = x.clone()
        b[b < 0] = 0
        a = self.quantizer_a(a)
        b = self.quantizer_b(b)
        x = a + b
        return x


class QGELU(nn.Module):
    def __init__(self,
                 quant=False,):
        super(QGELU, self).__init__()

        self.quant = quant

    def forward(self, x):
        if not self.quant:
            return nn.GELU()(x)
        relu6 = nn.ReLU6()
        x = x * relu6(1.702 * x + 3) / 6
        return x

class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        self.mode = 'int'

    def int_sqrt(self, x):
        output = torch.zeros_like(x)
        if (x < 0).any():
            raise ValueError('sqrt(x): x should great than zero')
        non_zero_mask = (x!=0)
        cur_sqrt_value = torch.ones_like(x)
        while True:
            pre = cur_sqrt_value
            cur_sqrt_value = (cur_sqrt_value + x / cur_sqrt_value) / 2
            if (torch.abs(cur_sqrt_value - pre) < 1e-6).all():
                break
        cur_sqrt_value *= non_zero_mask
        return (cur_sqrt_value)

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        if self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif self.mode == 'int':
            B,N,C = x.shape
            mu = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mu) / self.int_sqrt(var + self.eps)
            x = self.weight * x + self.bias
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scales):
        scales_all_none = True
        for i in scales:
            if i != None:
                scales_all_none = False
                break
        if self.log_i_softmax and not scales_all_none:
            softmax_out = []
            for i in range(len(scales)):
                scale = scales[i]
                head_wise_x = x[:, i, :, :]
                exp_int_headwise, exp_int_sum_headwise = self.int_softmax(head_wise_x, scale)
                softmax_out_headwise = exp_int_headwise / exp_int_sum_headwise
                softmax_out.append(softmax_out_headwise)
            softmax_out = torch.stack(softmax_out, dim=1)
            return  softmax_out
        else:
            x = x.softmax(dim=-1)
            return  x