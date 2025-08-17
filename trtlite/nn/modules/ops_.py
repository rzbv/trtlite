# implement modules which have not learnable parameters,
# these modules actually are functionals
import numpy as np
import tensorrt as trt
from tensorrt.tensorrt import DimsHW, PoolingType, Weights

from .. import functional as F
from .context import get_default_trt_context
from .module import Module


class MaxPool2d(Module):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, ) * 2
        self.stride = stride if isinstance(stride, tuple) else (stride, ) * 2
        self.padding = padding if isinstance(padding,
                                             tuple) else (padding, ) * 2
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        ctx = get_default_trt_context()
        # module_name = get_current_module_name()

        layer = ctx.network.add_pooling_nd(
            input=x,
            type=PoolingType.MAX,
            window_size=DimsHW(self.kernel_size),
        )
        layer.stride_nd = self.stride
        # layer.padding = self.padding
        layer.padding_nd = self.padding
        if self.ceil_mode:
            layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
        else:
            layer.padding_mode = trt.PaddingMode.SAME_UPPER

        return layer.get_output(0)


class AvgPool2d(Module):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, ) * 2
        self.stride = stride if isinstance(stride, tuple) else (stride, ) * 2
        if self.stride[0] is None:
            self.stride = self.kernel_size
        self.padding = padding if isinstance(padding,
                                             tuple) else (padding, ) * 2
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):
        return F.avg_pool(x, self.kernel_size, self.stride, self.padding,
                          self.ceil_mode, self.count_include_pad)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        h, w = x.shape[-2:]
        out_h = h if self.output_size[0] is None else self.output_size[0]
        out_w = w if self.output_size[1] is None else self.output_size[1]
        kernel_h = h - out_h + 1
        kernel_w = w - out_w + 1
        return F.avg_pool(x, (kernel_h, kernel_w), (1, 1), (0, 0))


class ZeroPad2d(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = [padding] * 4
        self.left_top = (padding[0], padding[2])
        self.right_bottom = (padding[1], padding[3])

    def forward(self, x):
        return F.padding(x, self.left_top, self.right_bottom)


class ReLU(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(x, alpha=self.negative_slope)


# TODO: use add deconv to implement this op for now
class Upsample(Module):
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = scale_factor
        self.align_corners = scale_factor

    def forward(self, x):
        ctx = get_default_trt_context()
        in_dim = 1 if ctx.is_dynamic_shape else 0
        in_channels = x.shape[in_dim]
        kernel_size = (int(self.scale_factor), ) * 2

        kval = np.ones(int(in_channels * self.scale_factor**2),
                       dtype=np.float32)
        kernel = Weights(kval)
        bias = Weights(np.zeros(in_channels, dtype=np.float32))

        layer = ctx.network.add_deconvolution(x, in_channels,
                                              DimsHW(kernel_size), kernel,
                                              bias)
        layer.stride = DimsHW(kernel_size)
        layer.num_groups = in_channels
        return layer.get_output(0)
