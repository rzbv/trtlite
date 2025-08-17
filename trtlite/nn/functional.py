from collections import OrderedDict
from functools import partial
from typing import List, Optional, Union

import numpy as np
import tensorrt as trt

from ..plugins.common import PluginCreatorContext
from .modules.context import get_default_trt_context
from .modules.module import get_current_module_name

# ## Type Definition ###
ITensor = trt.ITensor
Dims = trt.tensorrt.Dims
DimsHW = trt.tensorrt.DimsHW
IntOrTuple = Optional[Union[tuple, int]]
ArrOrTensor = Optional[Union[np.ndarray, ITensor]]


def get_trt_dim(ctx, xs, dim):
    dims = xs[0].shape if isinstance(xs, (list, tuple)) else xs.shape
    dims = list(dims)
    if dim < 0:
        dim_len = len(dims)
        dim += dim_len
    else:
        return dim
    
    return dim


def torch_dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask."""
    if not isinstance(dim, tuple):
        dim = (dim, )

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        # axes |= 1 << (d - 1)  # -1 to remove batch dimension
        axes |= 1 << d

    return axes


def softmax(x: ITensor, axis: int) -> ITensor:
    ctx = get_default_trt_context()
    layer = ctx.network.add_softmax(x)
    layer.set_axes(1 << axis)
    return layer.get_output(0)


def activation(x: ITensor,
               op: str,
               alpha: Optional[float] = None,
               beta: Optional[float] = None) -> ITensor:
    """
    Args:

      alpha (float): the alpha parameter, only used for
        LEAKY_RELU, ELU, SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH.
      beta (float): the beta parameter, only used for
        SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH.
    """

    OP_NAME_TYPE_MAP = {
        'relu': trt.ActivationType.RELU,
        'sigmoid': trt.ActivationType.SIGMOID,
        'tanh': trt.ActivationType.TANH,
        'elu': trt.ActivationType.ELU,
        'selu': trt.ActivationType.SELU,
        'softsign': trt.ActivationType.SOFTSIGN,
        'softplus': trt.ActivationType.SOFTPLUS,
        'clip': trt.ActivationType.CLIP,
        'hard_sigmoid': trt.ActivationType.HARD_SIGMOID,
        'scaled_tanh': trt.ActivationType.SCALED_TANH,
        'thresholded_relu': trt.ActivationType.THRESHOLDED_RELU,
        'leaky_relu': trt.ActivationType.LEAKY_RELU
    }

    assert op in OP_NAME_TYPE_MAP, f'{op} not supported'
    ctx = get_default_trt_context()
    module_name = get_current_module_name()
    layer = ctx.network.add_activation(x, OP_NAME_TYPE_MAP[op])
    tmp = ctx.get_uniq_name(module_name)
    layer.name = tmp
    if alpha is not None:
        layer.alpha = alpha
    if beta is not None:
        layer.beta = beta
    return layer.get_output(0)


def leaky_relu_dla(x: ITensor,
                   alpha: Optional[float] = None,
                   beta: Optional[float] = None) -> ITensor:
    ctx = get_default_trt_context()
    if ctx.dla_core < 0:
        return partial(activation, op='leaky_relu')(x, alpha=alpha, beta=beta)

    scalex = ctx.network.add_scale(x,
                                   trt.ScaleMode.UNIFORM,
                                   shift=np.zeros((1, ), np.float32),
                                   power=np.ones((1, ), np.float32),
                                   scale=0.01 * np.ones(
                                       (1, ), np.float32)).get_output(0)
    out = elementwise_op(x, scalex, 'max')
    return out


relu = partial(activation, op='relu')
sigmoid = partial(activation, op='sigmoid')
tanh = partial(activation, op='tanh')
elu = partial(activation, op='elu')
selu = partial(activation, op='selu')
softsign = partial(activation, op='softsign')
softplus = partial(activation, op='softplus')
clip = partial(activation, op='clip')
hard_sigmoid = partial(activation, op='hard_sigmoid')
scaled_tanh = partial(activation, op='scaled_tanh')
thresholded_relu = partial(activation, op='thresholded_relu')
# leaky_relu = partial(activation, op='leaky_relu')
leaky_relu = leaky_relu_dla


UOper = trt.tensorrt.UnaryOperation
UNARY_OP_NAME_TYPE_MAP = OrderedDict(
    (('exp', UOper.EXP), ('log', UOper.LOG), ('sqrt', UOper.SQRT),
     ('recip', UOper.RECIP), ('abs', UOper.ABS), ('neg', UOper.NEG),
     ('sin', UOper.SIN), ('cos', UOper.COS), ('tan', UOper.TAN),
     ('sinh', UOper.SINH), ('cosh', UOper.COSH), ('asin', UOper.ASIN),
     ('acos', UOper.ACOS), ('atan', UOper.ATAN), ('asinh', UOper.ASINH),
     ('acosh', UOper.ACOSH), ('atanh', UOper.ATANH), ('ceil', UOper.CEIL),
     ('floor', UOper.FLOOR), ('erf', UOper.ERF), ('not', UOper.NOT)))


def unary(x: ITensor, op: str):
    assert op in UNARY_OP_NAME_TYPE_MAP, f'{op} not supported'
    ctx = get_default_trt_context()
    layer = ctx.network.add_unary(x, UNARY_OP_NAME_TYPE_MAP[op])
    return layer.get_output(0)


# special name for abs and not
(exp, log, sqrt, recip, abs_, neg, sin, cos, tan, sinh, cosh, asin, acos, atan,
 asinh, acosh, atanh, ceil, floor, ref,
 not_) = map(lambda op: partial(unary, op=op), UNARY_OP_NAME_TYPE_MAP.keys())


EWOper = trt.tensorrt.ElementWiseOperation
EW_OP_NAME_TYPE_MAP = OrderedDict(
    (('sum', EWOper.SUM), ('prod', EWOper.PROD), ('max', EWOper.MAX),
     ('min', EWOper.MIN), ('sub', EWOper.SUB), ('div', EWOper.DIV),
     ('pow', EWOper.POW), ('floor_div', EWOper.FLOOR_DIV), ('and', EWOper.AND),
     ('or', EWOper.OR), ('xor', EWOper.XOR), ('equal', EWOper.EQUAL),
     ('greater', EWOper.GREATER), ('less', EWOper.LESS)))


def elementwise_op(a: ITensor,
                   b: ITensor,
                   op: str,
                   precision=None,
                   out_type=None):
    assert op in EW_OP_NAME_TYPE_MAP, f'{op} not supported'
    ctx = get_default_trt_context()
    layer = ctx.network.add_elementwise(a, b, EW_OP_NAME_TYPE_MAP[op])
    if precision:
        layer.precision = precision
    if out_type:
        layer.set_output_type(0, out_type)
    return layer.get_output(0)


# special name for abs and not
(sum_, prod_, max_, min_, sub_, div_, pow_, floor_div_, and_, or_, xor_,
 equal_, greater_, less_) = map(lambda op: partial(elementwise_op, op=op),
                                EW_OP_NAME_TYPE_MAP.keys())

REOper = trt.tensorrt.ReduceOperation
RE_OP_NAME_TYPE_MAP = OrderedDict(
    (('sum', REOper.SUM), ('prod', REOper.PROD), ('max', REOper.MAX),
     ('min', REOper.MIN), ('avg', REOper.AVG)))


def reduce_op(a: ITensor, op: str, dim: IntOrTuple, keep_dims: bool = False):
    def torch_dim_to_trt_axes(dim):
        """Converts torch dim, or tuple of dims to a tensorrt axes bitmask."""
        if not isinstance(dim, tuple):
            dim = (dim, )

        # create axes bitmask for reduce layer
        axes = 0
        for d in dim:
            axes |= 1 << (d - 1)  # -1 to remove batch dimension

        return axes

    assert op in RE_OP_NAME_TYPE_MAP, f'{op} not supported'
    ctx = get_default_trt_context()
    axis = torch_dim_to_trt_axes(dim)
    layer = ctx.network.add_reduce(a, RE_OP_NAME_TYPE_MAP[op], axis, keep_dims)
    return layer.get_output(0)


def avg_pool(input: ITensor,
             kernel_size: IntOrTuple = None,
             stride: IntOrTuple = None,
             padding: IntOrTuple = None,
             ceil_mode=False,
             count_include_pad=True):
    ctx = get_default_trt_context()

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if ctx.explicit_precision:
        module_name = get_current_module_name()
        scale = ctx.get_weight_by_name('_input_quantizer._amax',
                                       module_name,
                                       match=True)
        input = add_ops_dequantize(add_ops_quantize(input, scale), scale)

    layer = ctx.network.add_pooling_nd(input=input,
                                       type=trt.PoolingType.AVERAGE,
                                       window_size=kernel_size)

    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.average_count_excludes_padding = not count_include_pad

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


def padding_nd(x: ITensor, pre_padding: tuple, post_padding: tuple) -> ITensor:
    ctx = get_default_trt_context()
    layer = ctx.network.add_padding_nd(x, Dims(pre_padding),
                                       Dims(post_padding))
    return layer.get_output(0)


def padding(x: ITensor, pre_padding: tuple, post_padding: tuple) -> ITensor:
    ctx = get_default_trt_context()
    layer = ctx.network.add_padding_nd(x, DimsHW(pre_padding),
                                       DimsHW(post_padding))
    return layer.get_output(0)


def pad(input: ITensor, pad, mode='constant', value=0):
    assert len(pad) == 4, 'just support two dim padding currently'
    # pre_padding = (pad[0], pad[2])
    # post_padding = (pad[1], pad[3])
    pre_padding = (pad[2], pad[0])
    post_padding = (pad[3], pad[1])
    return padding(input, pre_padding, post_padding)


def cat(xs: List[ITensor], dim=0) -> ITensor:
    ctx = get_default_trt_context()
    # module_name = get_current_module_name()
    dim = get_trt_dim(ctx, xs, dim)

    shapes = [x.shape for x in xs]
    lens = [len(s) for s in shapes]
    assert len(set(lens)) == 1, shapes
    for ind, _shapes in enumerate(zip(*shapes)):
        if ind != dim:
            assert len(set(_shapes)) == 1, shapes

    layer = ctx.network.add_concatenation(xs)
    # layer.name = module_name + '.cat'
    layer.axis = dim
    return layer.get_output(0)


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None,
                recompute_scale_factor=None):
    layer_name = 'InterPolate_TRT'
    ctx = get_default_trt_context()
    creator = PluginCreatorContext.get_creator(layer_name)
    fn = creator.create_plugin(layer_name, trt.PluginFieldCollection([]))
    layer = ctx.network.add_plugin_v2([input], fn)
    return layer.get_output(0)


def get_shape(input):
    ctx = get_default_trt_context()
    layer = ctx.network.add_shape(input)
    return layer.get_output(0)


def squeeze(input, dim):
    ctx = get_default_trt_context()
    layer = ctx.network.add_shuffle(input)
    dims = list(input.shape)
    dim = get_trt_dim(ctx, input, dim)
    assert dims[dim] == 1, (f'can not squeeze {dim}, which is {dims[dim]}, '
                            f'input shape is {dims}')
    dims.pop(dim)
    layer.reshape_dims = dims
    return layer.get_output(0)


def unsqueeze(input, dim):
    ctx = get_default_trt_context()
    layer = ctx.network.add_shuffle(input)
    dims = list(input.shape)
    dim = get_trt_dim(ctx, input, dim)
    dims.insert(dim, 1)
    layer.reshape_dims = dims
    return layer.get_output(0)


def split(tensor, split_size_or_sections, dim=0):
    ctx = get_default_trt_context()
    size = split_size_or_sections
    dims = list(tensor.shape)
    dim = get_trt_dim(ctx, tensor, dim)
    if isinstance(size, int):
        size = [size]
    if sum(size) < dims[dim]:
        size.append(dims[dim] - sum(size))
    start = [
        0,
    ] * len(dims)
    stride = [1] * len(start)
    shape = dims
    outs = []
    offset = 0
    for sp in size:
        shape[dim] = sp
        start[dim] = offset
        offset += sp
        layer = ctx.network.add_slice(tensor,
                                      start=start,
                                      shape=shape,
                                      stride=stride)
        outs.append(layer.get_output(0))
    return tuple(outs)


def view(tensor, shape, name=None):
    # TODO: support -1
    # dims = list(tensor.shape)

    ctx = get_default_trt_context()
    layer = ctx.network.add_shuffle(tensor)
    if isinstance(shape, ITensor):
        layer.set_input(1, shape)
    else:
        layer.reshape_dims = tuple(shape)
    if name:
        layer.name = name
    return layer.get_output(0)

reshape = view

def permute(input, dims):
    ctx = get_default_trt_context()
    dims = [get_trt_dim(ctx, input, dim) for dim in dims]

    layer = ctx.network.add_shuffle(input)
    layer.second_transpose = tuple(dims)

    return layer.get_output(0)


def flatten(input, start_dim=0, end_dim=-1):
    ctx = get_default_trt_context()
    start_dim = get_trt_dim(ctx, input, start_dim)
    shape = list(input.shape)
    flat_len = 1
    for s in shape[start_dim:]:
        flat_len *= s
    new_shape = shape[:start_dim] + [
        flat_len,
    ]

    return view(input, new_shape)


def add(input, other, *, out=None):
    ctx = get_default_trt_context()
    module_name = get_current_module_name()

    shape = input.shape
    assert len(tuple(shape)) > 0, shape
    if not isinstance(other, ITensor):

        other = trt.Weights(np.ones(shape, dtype=np.float32) * other)
        layer = ctx.network.add_constant(shape, other)
        other = layer.get_output(0)

    # if len(input.shape) == 0 and len(other.shape) > 1:
    #     shape = (1, ) * len(other.shape)
    #     input = view(input, shape)
    if len(input.shape) > 1 and len(other.shape) == 0:
        shape = (1, ) * len(input.shape)
        other = view(other, shape)

    layer = ctx.network.add_elementwise(input, other,
                                        trt.ElementWiseOperation.SUM)
    layer.name = module_name + '.add'
    return layer.get_output(0)


def mul(input, other, *, out=None):
    # TODO:
    # assert len(input.shape) > len(other.shape), (input.shape, other.shape)
    ctx = get_default_trt_context()
    if not isinstance(other, ITensor):
        other = add_constant([1], other)

    if len(input.shape) != len(other.shape):
        shape = (1, ) * (len(input.shape) - len(other.shape)) + tuple(
            other.shape)
        other = view(other, shape)
    
    layer = ctx.network.add_elementwise(input, other,
                                        trt.ElementWiseOperation.PROD)
    return layer.get_output(0)


def max(input, dim=None, keepdim=False, *, out=None):
    ctx = get_default_trt_context()
    if dim is None:
        axes = sum(1 << i for i in range(len(input.shape)))
        layer = ctx.network.add_reduce(input, trt.ReduceOperation.MAX, axes, keepdim)
        return layer.get_output(0)
        
    else:
        layer = ctx.network.add_reduce(input, trt.ReduceOperation.MAX,
                                       torch_dim_to_trt_axes(dim), keepdim)
        return layer.get_output(0)


def gather(input, dim, index, *, sparse_grad=False, out=None):
    # refer https://zhuanlan.zhihu.com/p/352877584
    ctx = get_default_trt_context()
    if not isinstance(index, ITensor):
        if isinstance(index, (tuple, list)):
            index = np.array(index, dtype=np.int32)
        elif isinstance(index, int):
            index = np.array([index], dtype=np.int32)
        index = add_constant(index.shape, index, np.int32)
    
    layer = ctx.network.add_gather(input, index, dim)
    return layer.get_output(0)


def slice(input, shape, start=None, stride=None, dim=None):
    ctx = get_default_trt_context()
    is_dynamic = (-1 in input.shape) or isinstance(shape, ITensor)
    dims_len = len(input.shape)
    assert is_dynamic, 'slice only support dynamic shape'
    assert dim is not None, 'slice dim is required'

    if stride is None:
        stride = add_constant([dims_len], 1, np.int32)

    if start is None:
        start = add_constant([dims_len], 0, np.int32)
    else:
        # start: [0, 0, ..., start, 0, 0, ...]
        ts = [start]
        if dim > 0:
            ts.insert(0, add_constant([dim], 0, np.int32))

        if dim < dims_len - 1:
            ts.append(add_constant([dims_len - 1 - dim], 0, np.int32))

        start = cat(ts)

    in_shape = get_shape(input)
    ts = [shape]
    if dim > 0:
        part = ctx.network.add_slice(
            in_shape, start=[0], shape=[dim], stride=[1]).get_output(0)
        ts.insert(0, part)

    if dim < dims_len-1:
        part = ctx.network.add_slice(
            in_shape, start=[dim+1], shape=[dims_len-1-dim], stride=[1]).get_output(0)
        ts.append(part)

    shape = cat(ts)

    layer = ctx.network.add_slice(
        input, 
        trt.Dims([0,] * dims_len), 
        trt.Dims([0,] * dims_len), 
        trt.Dims([1,] * dims_len))
    
    # layer.set_input(1, start)
    layer.set_input(2, shape)
    # layer.set_input(3, stride)
    return layer.get_output(0)
  

def arange(length, dtype=np.int32):
    ctx = get_default_trt_context()
    if not isinstance(length, ITensor):
        length = add_constant([1], length, dtype)
    
    layer = ctx.network.add_fill(trt.Dims([3]), trt.FillOperation.LINSPACE)
    layer.set_input(0, length)
    layer.set_input(1, add_constant([], 0, dtype))
    layer.set_input(2, add_constant([1], 1, dtype))
    # layer.set_output_type(0, dtype) not working

    return layer.get_output(0)


def transpose(input, dims0, dims1=None):
    if isinstance(dims0, (tuple, list)):
        perm = dims0
    elif isinstance(dims0, int):
        assert isinstance(dims1, int)
        dim_len = len(input.shape)
        perm = np.arange(0, dim_len, dtype=np.int32).tolist()
        perm[dims0] = dims1
        perm[dims1] = dims0

    ctx = get_default_trt_context()
    layer = ctx.network.add_shuffle(input)
    layer.first_transpose = perm
    return layer.get_output(0)



def identity(input, dtype=None):
    ctx = get_default_trt_context()
    layer = ctx.network.add_identity(input)
    if dtype:
        layer.set_output_type(0, dtype)
    return layer.get_output(0)


def add_ops_quantize(input: ArrOrTensor,
                     scale: Union[float, np.ndarray],
                     input_shape=None,
                     name=''):
    ctx = get_default_trt_context()
    module_name = get_current_module_name()

    if isinstance(scale, float):
        scale = ctx.network.add_constant([1], np.array(scale, np.float32))
    else:
        if len(scale.ravel()) == 1:
            scale = ctx.network.add_constant([1],
                                             np.array(scale.ravel()[0],
                                                      dtype=np.float32))
        else:
            scale = np.array(scale.ravel(), dtype=np.float32)
            scale = ctx.network.add_constant(scale.shape, scale)
    scale.name = module_name + f'.q.{name}_scale'

    if isinstance(input, np.ndarray):
        input = np.array([input.ravel()], dtype=np.float32)
        input = ctx.network.add_constant(input_shape, input)
        input.name = module_name + f'.q_{name}.const'
        input = input.get_output(0)

    _ops = ctx.network.add_quantize(input=input, scale=scale.get_output(0))
    _ops.name = module_name + f'.q_{name}'
    _ops.axis = 0
    return _ops.get_output(0)


def add_ops_dequantize(input: ITensor,
                       scale: Union[float, np.ndarray],
                       name=''):
    ctx = get_default_trt_context()
    module_name = get_current_module_name()

    if isinstance(scale, float):
        scale = ctx.network.add_constant([1], np.array(scale, np.float32))
    else:
        if len(scale.ravel()) == 1:
            scale = ctx.network.add_constant([1],
                                             np.array(scale.ravel()[0],
                                                      dtype=np.float32))
        else:
            scale = np.array(scale.ravel(), dtype=np.float32)
            scale = ctx.network.add_constant(scale.shape, scale)
    scale.name = module_name + f'.dq.{name}_scale'
    _ops = ctx.network.add_dequantize(input=input, scale=scale.get_output(0))
    _ops.name = module_name + f'.dq_{name}'
    _ops.axis = 0
    return _ops.get_output(0)


def add_constant(shape, value, dtype=np.float32, name=None):
    ctx = get_default_trt_context()
    val = value
    if not isinstance(value, np.ndarray):
        val = np.full(shape, value, dtype=dtype)
    
    layer = ctx.network.add_constant(shape, val)
    if name:
        layer.name = name
    return layer.get_output(0)


def add_tile(input, reps, expand_dim=None, debug=False):
    ctx = get_default_trt_context()
    shape = get_shape(input)
    if expand_dim is not None:
        dims_len = len(input.shape)
        assert 0 <= expand_dim <= dims_len
        
        exp_shape = [shape]
        dst_shape = [shape]
        if expand_dim == 0:
            exp_shape.insert(0, add_constant([1], 1, np.int64))
            dst_shape.insert(0, reps)
        elif expand_dim == dims_len:
            exp_shape.append(add_constant([1], 1, np.int64))
            dst_shape.append(reps)

        exp_shape = cat(exp_shape)
        dst_shape = cat(dst_shape)
        input = view(input, exp_shape)
        
    else:
        assert reps.shape[0] == len(input.shape), (reps.shape, input.shape)
        dst_shape = prod_(shape, reps)

    sh_len = len(input.shape)
    layer = ctx.network.add_slice(
        input, 
        trt.Dims([0,] * sh_len), 
        trt.Dims([0,] * sh_len), 
        trt.Dims([1,] * sh_len))
    
    layer.set_input(2, dst_shape)
    layer.mode = trt.SampleMode.WRAP

    return layer.get_output(0)
    
