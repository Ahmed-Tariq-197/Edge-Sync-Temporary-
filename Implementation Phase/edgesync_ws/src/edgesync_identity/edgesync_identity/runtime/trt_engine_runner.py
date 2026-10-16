"""TensorRT 8.6 engine runner (static shapes, one input, one output, PyCUDA buffers)."""
import os
from typing import List, Tuple

import numpy as np

from .base_inference_runner import BaseInferenceRunner

WARMUP_ITERATIONS = 10
EXPECTED_INPUT_COUNT = 1
EXPECTED_OUTPUT_COUNT = 1


class TensorRTUnavailableError(RuntimeError):
    """Raised when TensorRT or PyCUDA cannot be imported on this host."""


class TrtEngineRunner(BaseInferenceRunner):
    """Loads a serialized engine once and keeps it resident for repeated inference."""

    def __init__(self, engine_path: str, warmup_iterations: int = WARMUP_ITERATIONS):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self._trt, self._cuda = self._import_runtime()
        self.engine_path = engine_path

        logger = self._trt.Logger(self._trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, self._trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = self._cuda.Stream()
        self._allocate_buffers()

        warmup_tensor = np.zeros(self.input_shape, dtype=self._input_dtype)
        for _ in range(warmup_iterations):
            self.infer(warmup_tensor)

    @staticmethod
    def _import_runtime():
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as exc:
            raise TensorRTUnavailableError(str(exc)) from exc
        return trt, cuda

    def _allocate_buffers(self) -> None:
        inputs: List[Tuple[int, tuple, np.dtype]] = []
        outputs: List[Tuple[int, tuple, np.dtype]] = []
        for index in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                raise ValueError(f"Dynamic shapes are not supported: binding {index} has {shape}")
            dtype = np.dtype(self._trt.nptype(self._engine.get_binding_dtype(index)))
            target = inputs if self._engine.binding_is_input(index) else outputs
            target.append((index, shape, dtype))

        if len(inputs) != EXPECTED_INPUT_COUNT or len(outputs) != EXPECTED_OUTPUT_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_INPUT_COUNT} input and {EXPECTED_OUTPUT_COUNT} output "
                f"binding, found {len(inputs)} and {len(outputs)}")

        _, self.input_shape, self._input_dtype = inputs[0]
        _, self.output_shape, self._output_dtype = outputs[0]
        input_size = int(self._trt.volume(self.input_shape))
        output_size = int(self._trt.volume(self.output_shape))

        self._host_input = self._cuda.pagelocked_empty(input_size, self._input_dtype)
        self._host_output = self._cuda.pagelocked_empty(output_size, self._output_dtype)
        self._device_input = self._cuda.mem_alloc(self._host_input.nbytes)
        self._device_output = self._cuda.mem_alloc(self._host_output.nbytes)

        bindings = [0] * self._engine.num_bindings
        bindings[inputs[0][0]] = int(self._device_input)
        bindings[outputs[0][0]] = int(self._device_output)
        self._bindings = bindings

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = np.ascontiguousarray(input_tensor, dtype=self._input_dtype)
        if tensor.shape != self.input_shape:
            raise ValueError(f"Input shape {tensor.shape} does not match engine {self.input_shape}")
        np.copyto(self._host_input, tensor.reshape(-1))
        self._cuda.memcpy_htod_async(self._device_input, self._host_input, self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._host_output, self._device_output, self._stream)
        self._stream.synchronize()
        return np.array(self._host_output, dtype=np.float32).reshape(-1)
