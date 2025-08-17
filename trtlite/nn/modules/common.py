import ctypes
import os

import numpy as np
import pycuda.autoinit  # noqa: F401 # Required for CUDA context initialization
import pycuda.driver as cuda
import tensorrt as trt

LIBCUDART = ctypes.cdll.LoadLibrary('libcudart.so')
LIBCUDART.cudaGetErrorString.restype = ctypes.c_char_p


def GiB(val):
    return val * 1 << 30


def cudaSetDevice(device_idx):
    ret = LIBCUDART.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = LIBCUDART.cudaGetErrorString(ret)
        raise RuntimeError('cudaSetDevice: ' + error_string)
    

class MemBuffer(object):
    def __init__(self, size, trt_type):
        self._size = size
        self._capacity = size
        self._trt_type = trt_type
        self._allocate()

    def _get_element_size(self, trt_type):
        mapping = {
            trt.DataType.INT32: 4,
            trt.DataType.FLOAT: 4,
            trt.DataType.HALF: 2,
            trt.DataType.INT8: 1,
        }
        assert trt_type in mapping
        return mapping.get(trt_type)

    def _allocate(self):
        self.host = cuda.pagelocked_empty(self._size,
                                          trt.nptype(self._trt_type))
        self.device = cuda.mem_alloc(self.nbytes())
        
    def size(self):
        return self._size

    def resize(self, new_size):
        assert new_size <= self._capacity

        if new_size <= self._capacity:
            self._size = new_size
        else:
            self._size = new_size
            self._capacity = new_size
            if self.device:
                self.device.free()

            self.device = cuda.mem_alloc(self.nbytes())

        self.host = None
        self.host = cuda.pagelocked_empty(self._size,
                                          trt.nptype(self._trt_type))

    def nbytes(self):
        return self._size * self._get_element_size(self._trt_type)

    def __del__(self):
        self.host = None
        if self.device:
            self.device.free()

        self.device = None


class InferenceSession(object):
    def __init__(self, trt_engine_path, device_id=0):
        cudaSetDevice(device_id)
        self.device_id = device_id
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, '') 
        self.trt_runtime = trt.Runtime(trt_logger)
        if not os.path.exists(trt_engine_path):
            raise Exception('Engine path not exists!')

        with open(trt_engine_path, 'rb') as f:
            self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())
        assert self.engine

        self.context = self.engine.create_execution_context()
        self.is_dynamic_shape = True

        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_attrs = []
        self.input_attrs = []

        self._init_bindings()
        self._pre_allocate_buffers()

    def _init_bindings(self):
        binding_infos = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            binding_infos.append((i, name, is_input))

        self.binding_infos = binding_infos
        
    def _pre_allocate_buffers(self, profile_idx=0):
        # update binding shape for input
        # todo: index and binding names may not be 1-1 orderly mapping
        input_dims = []
        for idx, name, is_input in self.binding_infos:
            if is_input:
                input_dims.append(
                    self.engine.get_tensor_profile_shape(name, profile_idx)[-1])

        input_idx = 0
        for _, name, is_input in self.binding_infos:
            if is_input:
                input_shape = input_dims[input_idx]
                self.context.set_input_shape(name, input_shape)
                input_idx += 1
            
        self.context.infer_shapes()
        assert self.context.all_binding_shapes_specified

        for idx, name, is_input in self.binding_infos:
            size = trt.volume(self.context.get_tensor_shape(name))
            trt_type = self.engine.get_tensor_dtype(name)

            buffer = MemBuffer(size, trt_type)
            # Append the device buffer to device bindings.
            self.bindings.append(int(buffer.device))

            # Append to the appropriate list.
            shape = self.context.get_tensor_shape(name)
            if is_input:
                self.inputs.append(buffer)
                self.input_attrs.append((name, shape))
            else:
                self.outputs.append(buffer)
                self.output_attrs.append((name, shape))

    def _dynamic_allocate_buffers(self, input_dims):
        # update binding shape for input
        for idx, (name, shape) in enumerate(self.input_attrs):
            input_shape = input_dims[idx]
            self.context.set_input_shape(name, input_shape)
        
        self.context.infer_shapes()
        assert self.context.all_binding_shapes_specified

        # update io buffers            
        for i, buf in enumerate(self.inputs):
            binding = self.input_attrs[i][0]
            shape = self.context.get_tensor_shape(binding)
            size = trt.volume(shape)
            self.input_attrs[i] = (binding, shape)
            buf.resize(size)

        for i, buf in enumerate(self.outputs):
            binding = self.output_attrs[i][0]
            shape = self.context.get_tensor_shape(binding)
            size = trt.volume(shape)
            self.output_attrs[i] = (binding, shape)
            buf.resize(size)

    def do_inference_v3(self):
        # Transfer input data to the GPU.
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        # Run inference.
        for i, binding in enumerate(self.bindings):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, binding)

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]
        # Synchronize the stream
        self.stream.synchronize()

    def _copy_to_pin_memory(self, xs):
        assert len(self.input_attrs) == len(xs), f'{len(self.input_attrs)} vs {len(xs)}'
        for i, x in enumerate(xs):
            xshape = x.shape
            assert xshape == self.input_attrs[i][1], (xshape, self.input_attrs[i][1])
            np.copyto(self.inputs[i].host, x.ravel())

    def run(self, input_dict):
        xs = [input_dict[name] for name, _ in self.input_attrs]
        input_dims = [x.shape for x in xs]
        self._dynamic_allocate_buffers(input_dims)
        self._copy_to_pin_memory(xs)
        self.do_inference_v3()

        result = []
        for output, (key, shape) in zip(self.outputs, self.output_attrs):
            result.append((key, output.host.reshape(shape)))
        return dict(result)
