import numpy as np
import tensorrt as trt
from tensorrt.tensorrt import ScaleMode, Weights

from .. import functional as F
from .context import get_default_trt_context
from .module import Module, get_current_module_name

ITensor = trt.ITensor


def get_scale(amax):
    return amax / 127


class Conv2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1,
                 bias=True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding_mode = None
        if padding == 'same':
            self.padding_mode = trt.PaddingMode.SAME_UPPER
        elif padding == 'valid':
            padding = 0
            
        self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                   padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,
                                                                      dilation)
        self.groups = groups
        self.bias = bias

    def forward(self, x):
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        # breakpoint()
        weight = ctx.get_weight_by_name('weight', module_name, match=True)
        bias = ctx.get_weight_by_name('bias', module_name,
                                      match=True) if self.bias else None
        if ctx.have_mask:
            weight_mask = ctx.get_weight_by_name('weight_mask',
                                                 module_name,
                                                 match=True)
            assert len(weight_mask.shape) == 4
            indexs = np.sum(weight_mask, axis=(1, 2, 3))
            self.out_channels = len(np.where(indexs > 0)[0])

        if False and ctx.explicit_precision:
            input_scale = get_scale(
                ctx.get_weight_by_name('_input_quantizer._amax',
                                       module_name,
                                       match=True))
            weight_scale = get_scale(
                ctx.get_weight_by_name('_weight_quantizer._amax',
                                       module_name,
                                       match=True))

            x = F.add_ops_dequantize(
                F.add_ops_quantize(x, input_scale, name='input'), input_scale,
                'input')
            weight = F.add_ops_dequantize(
                F.add_ops_quantize(weight, weight_scale,
                                   [self.out_channels, self.in_channels] +
                                   list(self.kernel_size), 'wei'),
                weight_scale, 'wei')
            layer = ctx.network.add_convolution_nd(
                input=x,
                num_output_maps=self.out_channels,
                kernel_shape=self.kernel_size,
                kernel=trt.Weights(),
                bias=bias)
            layer.set_input(1, weight)
        else:
            layer = ctx.network.add_convolution_nd(
                input=x,
                num_output_maps=self.out_channels,
                kernel_shape=self.kernel_size,
                kernel=weight,
                bias=bias)

        layer.name = module_name
        # layer.stride = self.stride
        layer.stride_nd = self.stride
        if self.padding_mode:
            layer.padding_mode = self.padding_mode
        else:
            # layer.padding = self.padding
            layer.padding_nd = self.padding

        # layer.dilation = self.dilation
        layer.dilation_nd = self.dilation
        
        if self.groups is not None:
            layer.num_groups = self.groups

        return layer.get_output(0)


class Conv1d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        # reshape to 2D
        layer = ctx.network.add_shuffle(x)
        layer.reshape_dims = (-1, x.shape[-1], 1)

        layer = ctx.network.add_convolution_nd(
            input=layer.get_output(0),
            num_output_maps=self.out_channels,
            kernel_shape=self.kernel_size,
            kernel=ctx.get_weight_by_name('Conv1d.weight', module_name),
            bias=ctx.get_weight_by_name('Conv1d.bias', module_name))


        layer.stride_nd = self.stride
        layer.padding_nd = self.padding
        layer.dilation_nd = self.dilation

        if self.groups is not None:
            layer.num_groups = self.groups

        return layer.get_output(0)


class Linear(Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.in_fea = in_features
        self.out_fea = out_features
        self.bias = bias

    def forward(self, x):
        assert len(x.shape) <= 4, 'Linear only support 2D, 3D, 4D input'
        
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        weight = ctx.get_weight_by_name('weight', module_name, match=True)
        bias = ctx.get_weight_by_name('bias', module_name,
                                      match=True) if self.bias else None
        
        # reshape to 2d
        layer = ctx.network.add_shuffle(x)
        layer.reshape_dims = (-1, x.shape[-1])

        weights = ctx.network.add_constant(
            (self.out_fea, self.in_fea), weight).get_output(0)
        if bias is not None:
            bias = ctx.network.add_constant((1, self.out_fea), bias).get_output(0)

        layer = ctx.network.add_matrix_multiply(
            layer.get_output(0), trt.MatrixOperation.NONE,
            weights, trt.MatrixOperation.TRANSPOSE
        )
        layer.name = module_name + '.linear'

        out = layer.get_output(0)
        if bias is not None:
            layer_bias = ctx.network.add_elementwise(   
                out,
                bias,
                op=trt.ElementWiseOperation.SUM
            )
            layer_bias.name = module_name + '.linear.bias'
            out = layer_bias.get_output(0)

        if len(x.shape) == 3:
            shape = F.get_shape(x)
            B = F.gather(shape, 0, 0)
            W = F.gather(shape, 0, 1)
            H = F.add_constant([1], self.out_fea, np.int64)
            out = F.reshape(out, F.cat([B, W, H]))
        elif len(x.shape) == 4:
            shape = F.get_shape(x)
            B = F.gather(shape, 0, 0)
            H = F.gather(shape, 0, 1)
            W = F.gather(shape, 0, 2)
            C = F.add_constant([1], self.out_fea, np.int64)
            out = F.reshape(out, F.cat([B, H, W, C]))

        return out


class BatchNorm2d(Module):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, x):
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        gamma = ctx.get_weight_by_name('weight', module_name, match=True)
        beta = ctx.get_weight_by_name('bias', module_name, match=True)
        mean = ctx.get_weight_by_name('running_mean', module_name, match=True)
        var = ctx.get_weight_by_name('running_var', module_name, match=True)

        scale = gamma / (np.sqrt(var + self.eps))
        shift = beta - mean * scale
        power = np.ones_like(scale)

        scale = Weights(scale)
        shift = Weights(shift)
        power = Weights(power)

        layer = ctx.network.add_scale(x, ScaleMode.CHANNEL, shift, scale,
                                      power)
        layer.name = module_name

        return layer.get_output(0)


class BatchNorm1d(BatchNorm2d):
    def forward(self, x):
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        gamma = ctx.get_weight_by_name('weight', module_name, match=True)
        beta = ctx.get_weight_by_name('bias', module_name, match=True)
        mean = ctx.get_weight_by_name('running_mean', module_name, match=True)
        var = ctx.get_weight_by_name('running_var', module_name, match=True)

        scale = gamma / (np.sqrt(var + self.eps))
        shift = beta - mean * scale
        power = np.ones_like(scale)

        scale = Weights(scale)
        shift = Weights(shift)
        power = Weights(power)

        # reshape to 2D
        # print('bn: ', x.shape)
        layer = ctx.network.add_shuffle(x)
        shape_len = len(x.shape)
        if ctx.is_dynamic_shape:
            shape_len -= 1

        if shape_len == 1:
            layer.reshape_dims = tuple(x.shape) + (1, 1)
        else:
            layer.reshape_dims = tuple(x.shape) + (1, )

        layer = ctx.network.add_scale(layer.get_output(0), ScaleMode.CHANNEL,
                                      shift, scale, power)

        # reshape back to 1D
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = tuple(x.shape)

        return layer.get_output(0)


# TODO: not finish yet.
class ConvTranspose2d(Module):
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros'):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size


class TensorQuantizer(Module):
    def forward(self, x):
        ctx = get_default_trt_context()
        module_name = get_current_module_name()

        scale = get_scale(
            ctx.get_weight_by_name('_amax', module_name, match=True))
        x = F.add_ops_dequantize(F.add_ops_quantize(x, scale), scale)
        return x
