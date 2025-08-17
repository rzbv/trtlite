import os

import pycuda.driver as cuda
import tensorrt as trt


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self,
                 training_data,
                 cache_file,
                 batch_size=64,
                 img_size=(640, 640, 3),
                 nbytes=4,
                 img_ext='jpg',
                 max_sample=None):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size
        # will be copied to the device and returned.
        if max_sample is not None and max_sample > 0:
            self.data_files = self.list_data(training_data,
                                             img_ext)[:max_sample]
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        mem_size = img_size[0] * img_size[1] * img_size[2] * nbytes
        self.device_input = cuda.mem_alloc(mem_size * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def list_data(self, training_data, img_ext):
        return [
            os.path.join(training_data, f) for f in os.listdir(training_data)
            if f.endswith(img_ext)
        ]

    def load_data(self, files):
        raise NotImplementedError

    # TensorRT passes along the names of the engine bindings to
    # the get_batch function.
    # You don't necessarily have to use them, but they can be useful
    # to understand the order of the inputs.
    # The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data_files):
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print('Calibrating batch {:}, containing {:} images'.format(
                current_batch, self.batch_size))

        batch = self.load_data(
            self.data_files[self.current_index:self.current_index +
                            self.batch_size]).ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
