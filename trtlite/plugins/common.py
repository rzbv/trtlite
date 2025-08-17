import ctypes
import os

import numpy as np
import tensorrt as trt


def get_tensorrt_op_path():
    """Get TensorRT plugins library path."""
    wildcard = '/usr/local/trtlite/lib/libtrtlite_plugin.so'
    return wildcard

plugin_is_loaded = False


def is_tensorrt_plugin_loaded():
    """Check if TensorRT plugins library is loaded or not.

    Returns:
        bool: plugin_is_loaded flag
    """
    global plugin_is_loaded
    return plugin_is_loaded


def load_tensorrt_plugin():
    """load TensorRT plugins library."""
    global plugin_is_loaded
    lib_path = get_tensorrt_op_path()
    if (not plugin_is_loaded) and os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        plugin_is_loaded = True


class PluginCreatorContext(object):
    REGISTRY = None

    @staticmethod
    def global_init(debug=False):
        if PluginCreatorContext.REGISTRY is None:
            ctypes.CDLL(get_tensorrt_op_path(), mode=ctypes.RTLD_GLOBAL)
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(TRT_LOGGER, '')
            PluginCreatorContext.REGISTRY = trt.get_plugin_registry()
            if debug:
                for creator in PluginCreatorContext.REGISTRY.plugin_creator_list:
                    print(creator.name)

    @staticmethod
    def get_creator(name, ver='1'):
        return PluginCreatorContext.REGISTRY.get_plugin_creator(name, ver, '')


def create_plugin_field(name, value, dtype):
    NP_TYPE_TO_TRT_TYPE_MAP = {
        np.int32: trt.PluginFieldType.INT32,
        np.float32: trt.PluginFieldType.FLOAT32,
        np.float16: trt.PluginFieldType.FLOAT16,
        np.int8: trt.PluginFieldType.INT8,
        np.int16: trt.PluginFieldType.INT16,
        np.int32: trt.PluginFieldType.INT32,
        np.char: trt.PluginFieldType.CHAR
    }

    if isinstance(dtype, str):
        return trt.PluginField(name, value, trt.PluginFieldType.DIMS)
    else:
        return trt.PluginField(name, np.array(value, dtype),
                               NP_TYPE_TO_TRT_TYPE_MAP[dtype])
