"""Static-shape TensorRT named-I/O runtime using one CUDA stream.

TensorRT and PyCUDA are imported only when a runner is constructed. The retained
CUDA primary context is compatible with other primary-context users, including
PyTorch. A runner is synchronous, single-threaded and not reentrant.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class TensorRTRunner:
    """Own an engine, its CUDA context and reusable pinned/device buffers."""

    def __init__(self, engine_path: str | Path, device: int = 0):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as error:
            raise RuntimeError("TensorRT and PyCUDA are required on the target Jetson") from error

        self._thread = threading.get_ident()
        self._cuda = cuda
        self._primary = None
        self._stream = None
        self._runtime = self._engine = self._context = None
        self._buffers = {}
        self.input_specs = {}
        self.output_specs = {}
        self._closed = False
        self.tensorrt_version = trt.__version__
        cuda.init()
        self._primary = cuda.Device(device).retain_primary_context()
        self._primary.push()
        try:
            self.device_name = cuda.Device(device).name()
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the engine; rebuild it on this device")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            self._stream = cuda.Stream()
            for index in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(index)
                shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
                if not shape or any(value <= 0 for value in shape):
                    raise ValueError(f"Only static nonempty tensor shapes are supported: {name} {shape}")
                if self._engine.is_shape_inference_io(name):
                    raise ValueError(f"Shape-inference tensors are not supported: {name}")
                if self._engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError(f"Only device I/O tensors are supported: {name}")
                if self._engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                    raise ValueError(f"Only linear I/O tensor formats are supported: {name}")
                dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
                if dtype.hasobject:
                    raise ValueError(f"Unsupported I/O dtype for {name}: {dtype}")
                spec = {"shape": shape, "dtype": dtype}
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_specs[name] = spec
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_specs[name] = spec
                else:
                    raise ValueError(f"Unknown tensor I/O mode for {name}")
                host = cuda.pagelocked_empty(shape, dtype)
                allocation = cuda.mem_alloc(host.nbytes)
                self._buffers[name] = {"host": host, "device": allocation}
                if not self._context.set_tensor_address(name, int(allocation)):
                    raise RuntimeError(f"TensorRT rejected the buffer address for {name}")
            if not self.input_specs or not self.output_specs:
                raise ValueError("The engine must have at least one input and one output")
        except BaseException:
            self._release_resources()
            raise
        finally:
            self._primary.pop()
            if self._closed:
                self._primary.detach()
                self._primary = None

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Use and close a TensorRTRunner on its creating thread")

    def infer(self, inputs: dict[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
        """Run H2D, execution and D2H in order; return independent output copies."""
        self._check_thread()
        if self._closed:
            raise RuntimeError("The TensorRT runner is closed")
        if not isinstance(inputs, dict):
            if len(self.input_specs) != 1:
                raise ValueError("Pass a named input dictionary for engines with multiple inputs")
            inputs = {next(iter(self.input_specs)): inputs}
        if set(inputs) != set(self.input_specs):
            raise ValueError(f"Expected input names {sorted(self.input_specs)}, got {sorted(inputs)}")
        for name, spec in self.input_specs.items():
            value = np.asarray(inputs[name])
            if value.shape != spec["shape"] or value.dtype != spec["dtype"]:
                raise ValueError(f"{name}: expected {spec['shape']} {spec['dtype']}, "
                                 f"got {value.shape} {value.dtype}")
            np.copyto(self._buffers[name]["host"], value, casting="no")
        self._primary.push()
        try:
            for name in self.input_specs:
                buffer = self._buffers[name]
                self._cuda.memcpy_htod_async(buffer["device"], buffer["host"], self._stream)
            if not self._context.execute_async_v3(stream_handle=self._stream.handle):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for name in self.output_specs:
                buffer = self._buffers[name]
                self._cuda.memcpy_dtoh_async(buffer["host"], buffer["device"], self._stream)
            self._stream.synchronize()
            return {name: self._buffers[name]["host"].copy() for name in self.output_specs}
        except BaseException:
            # Complete any queued work before host buffers may be reused or freed.
            self._stream.synchronize()
            raise
        finally:
            self._primary.pop()

    def _release_resources(self) -> None:
        """Release CUDA-owned resources while the primary context is current."""
        try:
            if self._stream is not None:
                self._stream.synchronize()
        finally:
            self._context = None
            for buffer in self._buffers.values():
                buffer["device"].free()
            self._buffers.clear()
            self._stream = None
            self._engine = self._runtime = None
            self._closed = True

    def close(self) -> None:
        """Release owned resources; repeated calls are harmless."""
        self._check_thread()
        if self._closed:
            return
        self._primary.push()
        try:
            self._release_resources()
        finally:
            self._primary.pop()
            self._primary.detach()
            self._primary = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# Keep a descriptive alias for validation and other static-engine clients.
TensorRTStaticRunner = TensorRTRunner
