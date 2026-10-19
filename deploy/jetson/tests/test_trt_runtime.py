"""Exercise actual runner control flow with asynchronous CUDA/TensorRT doubles."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trt_runtime import TensorRTRunner


class FakeBackend:
    def __init__(self, fail_execute=False, dynamic=False):
        self.events = []
        self.allocations = {}
        self.pinned = set()
        self.addresses = {}
        self.depth = 0
        self.fail_execute = fail_execute
        outer = self

        class Primary:
            def push(self):
                outer.depth += 1

            def pop(self):
                outer.depth -= 1

            def detach(self):
                outer.events.append(("detach", None))

        class Device:
            def __init__(self, index):
                self.index = index

            def retain_primary_context(self):
                return Primary()

            def name(self):
                return "test device"

        class Allocation:
            def __init__(self, size):
                self.pointer = len(outer.allocations) + 1
                self.array = None
                outer.allocations[self.pointer] = self

            def __int__(self):
                return self.pointer

            def free(self):
                if outer.depth != 1:
                    raise AssertionError("Device memory must be freed in the retained context")
                outer.events.append(("free", self.pointer))

        class Stream:
            handle = 123

            def synchronize(self):
                outer.events.append(("sync", self.handle))
                for operation in outer.pending:
                    operation()
                outer.pending.clear()

        class Context:
            def set_tensor_address(self, name, pointer):
                outer.addresses[name] = pointer
                return True

            def execute_async_v3(self, stream_handle):
                outer.events.append(("execute", stream_handle))
                if outer.fail_execute:
                    return False

                def execute():
                    value = outer.allocations[outer.addresses["images"]].array
                    outer.allocations[outer.addresses["probability"]].array = value[:, :1] + 1
                    outer.allocations[outer.addresses["auxiliary"]].array = value[:, :1] + 2

                outer.pending.append(execute)
                return True

        class Engine:
            num_io_tensors = 3

            def create_execution_context(self):
                return Context()

            def get_tensor_name(self, index):
                return ["images", "probability", "auxiliary"][index]

            def get_tensor_shape(self, name):
                return (1, 3, -1 if dynamic else 2, 2) if name == "images" else (1, 1, 2, 2)

            def get_tensor_dtype(self, name):
                return np.float32

            def get_tensor_mode(self, name):
                return "input" if name == "images" else "output"

            def is_shape_inference_io(self, name):
                return False

            def get_tensor_location(self, name):
                return "device"

            def get_tensor_format(self, name):
                return "linear"

        class Runtime:
            def __init__(self, logger):
                pass

            def deserialize_cuda_engine(self, data):
                return Engine()

        class Logger:
            WARNING = 1

            def __init__(self, severity):
                pass

        def pinned_empty(shape, dtype):
            array = np.empty(shape, dtype)
            outer.pinned.add(id(array))
            return array

        def htod(destination, source, stream):
            if id(source) not in outer.pinned:
                raise AssertionError("Async H2D requires the pinned host buffer")
            outer.events.append(("h2d", stream.handle))
            outer.pending.append(lambda: setattr(destination, "array", source.copy()))

        def dtoh(destination, source, stream):
            if id(destination) not in outer.pinned:
                raise AssertionError("Async D2H requires the pinned host buffer")
            outer.events.append(("d2h", stream.handle))
            outer.pending.append(lambda: np.copyto(destination, source.array))

        self.pending = []
        driver = types.ModuleType("pycuda.driver")
        driver.init = lambda: None
        driver.Device = Device
        driver.Stream = Stream
        driver.mem_alloc = Allocation
        driver.pagelocked_empty = pinned_empty
        driver.memcpy_htod_async = htod
        driver.memcpy_dtoh_async = dtoh
        pycuda = types.ModuleType("pycuda")
        pycuda.driver = driver
        trt = types.ModuleType("tensorrt")
        trt.__version__ = "10.test"
        trt.Logger = Logger
        trt.Runtime = Runtime
        trt.nptype = lambda dtype: dtype
        trt.TensorIOMode = types.SimpleNamespace(INPUT="input", OUTPUT="output")
        trt.TensorLocation = types.SimpleNamespace(DEVICE="device")
        trt.TensorFormat = types.SimpleNamespace(LINEAR="linear")
        self.modules = {"tensorrt": trt, "pycuda": pycuda, "pycuda.driver": driver}


class RuntimeTests(unittest.TestCase):
    def test_order_outputs_noncontiguous_input_and_owned_copies(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, backend.modules):
            engine = Path(temporary) / "test.trt"
            engine.write_bytes(b"test engine")
            with TensorRTRunner(engine) as runner:
                array = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)[:, :, :, ::-1]
                first = runner.infer({"images": array})
                np.testing.assert_array_equal(first["probability"], array[:, :1] + 1)
                np.testing.assert_array_equal(first["auxiliary"], array[:, :1] + 2)
                second = runner.infer(np.zeros_like(array))
                np.testing.assert_array_equal(first["probability"], array[:, :1] + 1)
                np.testing.assert_array_equal(second["probability"], np.ones((1, 1, 2, 2)))
                with self.assertRaisesRegex(ValueError, "expected"):
                    runner.infer(array.astype(np.float16))
                with self.assertRaisesRegex(ValueError, "input names"):
                    runner.infer({"wrong": array})
            runner.close()
            operations = [event for event in backend.events if event[0] in {"h2d", "execute", "d2h", "sync"}]
            self.assertEqual(operations[:5], [("h2d", 123), ("execute", 123),
                                             ("d2h", 123), ("d2h", 123), ("sync", 123)])
            self.assertEqual(backend.depth, 0)
            self.assertEqual(sum(event[0] == "free" for event in backend.events), 3)
            self.assertEqual(sum(event[0] == "detach" for event in backend.events), 1)

    def test_failed_execution_synchronizes_and_dynamic_engine_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary) / "test.trt"
            engine.write_bytes(b"test engine")
            backend = FakeBackend(fail_execute=True)
            with patch.dict(sys.modules, backend.modules):
                with TensorRTRunner(engine) as runner:
                    with self.assertRaisesRegex(RuntimeError, "execute_async_v3 failed"):
                        runner.infer(np.zeros((1, 3, 2, 2), dtype=np.float32))
                self.assertEqual(backend.depth, 0)
                self.assertIn(("sync", 123), backend.events)
            backend = FakeBackend(dynamic=True)
            with patch.dict(sys.modules, backend.modules):
                with self.assertRaisesRegex(ValueError, "static"):
                    TensorRTRunner(engine)
                self.assertEqual(backend.depth, 0)
                self.assertEqual(sum(event[0] == "detach" for event in backend.events), 1)


if __name__ == "__main__":
    unittest.main()
