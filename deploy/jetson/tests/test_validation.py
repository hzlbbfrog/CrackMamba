"""Regression tests for frozen-scan rejection and the engine build gate."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_export import IMAGE_DEPENDENCY, _inspect_loops, compare_probabilities


def value(name):
    return SimpleNamespace(name=name)


def node(operation, inputs, outputs, attributes=None):
    return SimpleNamespace(op_type=operation, input=inputs, output=outputs, name="", attribute=attributes or [])


def graph(nodes, inputs=(), outputs=()):
    return SimpleNamespace(node=nodes, input=[value(name) for name in inputs], output=[value(name) for name in outputs], initializer=[])


def scan_loop(frozen=False, shape_only=False):
    prefix = [node("Shape" if shape_only else "Identity", ["images"], ["scan_values"])]
    recurrence_source = "constant" if frozen else "scan_values"
    body = graph(
        [node("Add", ["state", recurrence_source], ["next_state"]), node("Identity", ["condition"], ["next_condition"])],
        inputs=("iteration", "condition", "state"),
        outputs=("next_condition", "next_state"),
    )
    loop = node("Loop", ["length", "continue", "initial_state"], ["final_state"], [SimpleNamespace(type=5, g=body)])
    return graph(prefix + [loop], inputs=("images",), outputs=("final_state",))


class ValidationTests(unittest.TestCase):
    def test_scan_state_keeps_image_value_dependency(self):
        result = _inspect_loops(scan_loop(), {"images": {IMAGE_DEPENDENCY}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["input_dependent_states"], ["state"])

    def test_frozen_scan_rejected_even_when_outer_graph_depends_on_image(self):
        with self.assertRaisesRegex(ValueError, "image values"):
            _inspect_loops(scan_loop(frozen=True), {"images": {IMAGE_DEPENDENCY}})

    def test_shape_dependency_is_not_image_value_dependency(self):
        with self.assertRaisesRegex(ValueError, "image values"):
            _inspect_loops(scan_loop(shape_only=True), {"images": {IMAGE_DEPENDENCY}})

    def test_mask_difference_can_fail_with_small_probability_error(self):
        reference = np.array([[[[0.49999]]]], dtype=np.float32)
        candidate = np.array([[[[0.50001]]]], dtype=np.float32)
        self.assertFalse(compare_probabilities(reference, candidate, 1e-3, 1e-3, 0)["passed"])

    def test_numerical_error_can_fail_when_masks_agree(self):
        reference = np.array([[[[0.1]]]], dtype=np.float32)
        candidate = np.array([[[[0.2]]]], dtype=np.float32)
        self.assertFalse(compare_probabilities(reference, candidate, 1e-4, 1e-3, 0)["passed"])

    def test_nonfinite_and_invalid_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_probabilities(np.zeros((1, 1, 1, 1)), np.full((1, 1, 1, 1), np.nan), 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare_probabilities(np.zeros((1, 1, 1, 1)), np.zeros((1, 2, 1, 1)), 0, 0, 0)

    def test_build_gate_rejects_stale_and_empty_reports(self):
        script = (Path(__file__).resolve().parents[1] / "build_engine.sh").read_text(encoding="utf-8")
        gate = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory)
            onnx_path = location / "model.onnx"
            checkpoint = location / "model.pth"
            report_path = location / "validation.json"
            onnx_path.write_bytes(b"verified graph")
            checkpoint.write_bytes(b"weights")
            report = {
                "schema_version": 1,
                "status": "passed",
                "graph_inspection": {"status": "passed", "scan_loops": [{}] * 8},
                "numeric_validation": {"samples": 1, "results": [{"onnx": {"passed": True}, "image_sha256": "image"}]},
                "onnx": {"sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()},
                "checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            command = [sys.executable, "-c", gate, str(onnx_path), str(checkpoint), str(report_path)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            onnx_path.write_bytes(b"modified graph")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            report["onnx"]["sha256"] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            report["numeric_validation"] = {"samples": 0, "results": []}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_build_requires_tensor_rt_10_version(self):
        script = (Path(__file__).resolve().parents[1] / "build_engine.sh").read_text(encoding="utf-8")
        version_check = script.split("<<'PY'\n")[2].split("\nPY\n", 1)[0]
        for version in ("TensorRT v100300", "TensorRT version: 10.3.0"):
            command = [sys.executable, "-c", version_check, version]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        for version in ("TensorRT v110000", "TensorRT version: 8.6.1", "Unknown version"):
            command = [sys.executable, "-c", version_check, version]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
