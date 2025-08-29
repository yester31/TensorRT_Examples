#  by yhpark 2025-8-25
# implicit quantization (PTQ) TensorRT example
import tensorrt as trt
import numpy as np
from cuda import cuda, cudart
from common_runtime import *
import os

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.bindingMemory = None
        self.batch_size = None
        self.calib_datas = None
        self.calib_count = None
        self.count = 0

    def set_calibrator(self, batch_size, shape, dtype, calib_data_path):
        self.batch_size = batch_size
        size = int(np.dtype(dtype).itemsize * np.prod(shape))
        self.batch_allocation  = cuda_call(cudart.cudaMalloc(size))
        self.calib_datas = np.load(calib_data_path)
        self.calib_count = self.calib_datas.shape[0]

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.batch_size:
            return self.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        try:
            if self.count == 0:
                print(f"Start calibration... (calib_count:{self.calib_count})")
            if self.calib_count - 1 == self.count:
                print("Finished calibration batches")
                return None

            tensor = self.calib_datas[self.count]
            self.count += 1
            batch = np.array(tensor, dtype=np.float32, order="C")
            memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]

        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            print("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)