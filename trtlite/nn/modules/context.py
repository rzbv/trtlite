import contextlib
import os
import pickle
import threading
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from .common import cudaSetDevice

TRT_VER = int(trt.__version__.split('.', 1)[0])

DTYPE = dict(
    fp32=trt.float32,
    fp16=trt.float32,
    # input not suport int8
    int8=trt.float32,
)


class TrtContext(object):

    have_mask = False

    def __init__(self,
                 weights_map_file=None,
                 masks_file=None,
                 is_dynamic_shape=False,
                 precision='fp32',
                 explicit_precision=False,
                 calib=None,
                 log_level=trt.Logger.INFO,
                 device_id=0,
                 dla_core=-1,
                 strict_type=False,
                 max_workspace_size=1 << 20):
        assert precision in ['fp32', 'fp16', 'int8']
        cudaSetDevice(device_id)
        self.device_id = device_id
        self.dla_core = dla_core
        self.strict_type = strict_type  # or explicit_precision
        self.explicit_precision = explicit_precision
        self.weights_map_file = weights_map_file
        self.masks_file = masks_file
        self.is_dynamic_shape = is_dynamic_shape
        if self.explicit_precision:
            self.precision = 'int8'
        else:
            self.precision = precision
        self.calib = calib
        if log_level == 'i':
            self.log_level = trt.Logger.INFO
        elif log_level == 'v':
            self.log_level = trt.Logger.VERBOSE
        else:
            self.log_level = log_level
        self.max_workspace_size = max_workspace_size
        self.init = False
        if self.precision == 'int8' and not self.explicit_precision:
            assert self.calib is not None, 'need calibrator when int8'
        
        self.layer_names = []

    def add_inputs(self, input_names, input_dims):
        assert self.init, 'TrtContext has not initialized'
        dtype = DTYPE[self.precision]
        self.input_tensors = []
        self.input_names = input_names
        for name, dim in zip(input_names, input_dims):
            tensor = self.network.add_input(name=name,
                                            shape=trt.Dims(dim),
                                            dtype=dtype)
            self.input_tensors.append(tensor)

        return self.input_tensors

    def mark_outputs(self, tensors, names=None):
        assert self.init, 'TrtContext has not initialized'
        if names is None:
            names = [f'output_{i}' for i in range(len(tensors))]

        self.output_names = names
        assert len(tensors) == len(names), (len(tensors), len(names))
        for tensor, name in zip(tensors, names):
            tensor.name = name
            self.network.mark_output(tensor)

    def set_optimization_profiles(self, input_names, profiles_dims):
        profile = self.builder.create_optimization_profile()
        for input_name, profile_dims in zip(input_names, profiles_dims):
            min_dims = trt.Dims(profile_dims[0])
            opt_dims = trt.Dims(profile_dims[1])
            max_dims = trt.Dims(profile_dims[2])
            profile.set_shape(input_name, min_dims, opt_dims, max_dims)
        self.config.add_optimization_profile(profile)

    def initialize(self):
        self.trt_logger = trt.Logger(self.log_level)
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
        if self.strict_type:
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if self.precision == 'fp16':
            self.config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = self.calib

        if TRT_VER >= 8 and self.dla_core >= 0:
            self.config.default_device_type = trt.DeviceType.DLA
            self.config.DLA_core = self.dla_core
            self.config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            print('Using DLA core %d.' % self.dla_core)

        flag = 0
        self.network = self.builder.create_network(flag)

        self.weight_map = TrtContext.load_weights_map(self.weights_map_file,
                                                      self.masks_file)
        self.init = True

    def build_engine(self, engine_file=None):
        self.trt_logger.log(trt.Logger.INFO, 'start to build engine')

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create engine")

        self.trt_logger.log(trt.Logger.INFO, 'succ to build engine')
        if engine_file is not None:
            with open(engine_file, 'wb') as f:
                f.write(engine_bytes)

    @classmethod
    def load_weights_map(cls, file, mask=None):
        weights = OrderedDict()
        if file is None:
            return weights
        assert os.path.exists(file), 'Unable to load weight file.'

        with open(file, 'rb') as f:
            weights = pickle.load(f)

        if mask and os.path.exists(mask):
            cls.have_mask = True
            with open(mask, 'rb') as f:
                masks = pickle.load(f)

        for key in list(weights.keys()):
            weights[key] = np.array([weights[key].ravel()], dtype=np.float32)
            if cls.have_mask:
                weights[key + '_mask'] = np.array(
                    masks[key], dtype=np.float32) if key in masks else None

        return weights

    @classmethod
    def save_weights_map(cls, state_dict, file):
        with open(file, 'wb') as f:
            pickle.dump(state_dict, f)

    def get_weight_by_name(self,
                           name,
                           module_name='',
                           match=False) -> np.ndarray:
        if False:
            all_keys = self.weight_map.keys()
            match_res_1 = [k for k in all_keys if k.startswith(module_name)]
            match_res_2 = [k for k in all_keys if k.endswith(name)]

            match_res = list(set(match_res_1).intersection(set(match_res_2)))
            assert len(match_res) == 1, (
                f'expect to get {module_name+"*"+name}, but get {match_res}',
                f'potential keys is {match_res_1}')
            weight_name = match_res[0]
        else:
            weight_name = name if len(
                module_name) == 0 else f'{module_name}.{name}'
        assert weight_name in self.weight_map, f'{weight_name} not exists'
        return self.weight_map.get(weight_name)

    def get_uniq_name(self, expect_name):
        if expect_name not in self.layer_names:
            self.layer_names.append(expect_name)
            return expect_name
        
        last_char = expect_name[-1]
        if last_char.isdigit():
            expect_name = expect_name[:-1] + str(int(last_char) + 1)
        else:
            expect_name = expect_name + '1'

        return self.get_uniq_name(expect_name)

class _DefaultStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""
    def __init__(self):
        super(_DefaultStack, self).__init__()
        self._enforce_nesting = True
        self.stack = []

    def get_default(self):
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @property
    def enforce_nesting(self):
        return self._enforce_nesting

    @enforce_nesting.setter
    def enforce_nesting(self, value):
        self._enforce_nesting = value

    @contextlib.contextmanager
    def get_controller(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            # stack may be empty if reset() was called
            if self.stack:
                if self._enforce_nesting:
                    if self.stack[-1] is not default:
                        raise AssertionError(
                            'Nesting violated for default stack of %s objects'
                            % type(default))
                    self.stack.pop()
                else:
                    self.stack.remove(default)


_default_trt_context_stack = _DefaultStack()  # pylint: disable=protected-access # noqa


def DefaultTrtContext(context):
    return _default_trt_context_stack.get_controller(context)


def get_default_trt_context():
    return _default_trt_context_stack.get_default()
