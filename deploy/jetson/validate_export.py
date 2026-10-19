"""Validate selective-scan structure and numerical agreement before deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IMAGE_DEPENDENCY = "__image_values__"


def sha256_file(path):
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_dependencies(graph, inherited, input_dependencies=None):
    """Trace tensor values, excluding dependence only on static tensor shapes."""
    dependencies = dict(inherited)
    for initializer in graph.initializer:
        dependencies[initializer.name] = set()
    for value in graph.input:
        dependencies[value.name] = set((input_dependencies or {}).get(value.name, set()))
    for node in graph.node:
        sources = set().union(*(dependencies.get(name, set()) for name in node.input))
        if node.op_type in {"Constant", "Shape", "Size"}:
            sources = set()
        # A Loop can capture tensors from its parent graph without listing them
        # as explicit node inputs. Its body must be inspected in that scope.
        for attribute in node.attribute:
            if attribute.type == 5:  # ONNX AttributeProto.GRAPH
                nested = attribute.g
                local = {}
                if node.op_type == "Loop":
                    for index, value in enumerate(nested.input):
                        if index == 0:
                            local[value.name] = set()
                        else:
                            name = node.input[index] if index < len(node.input) else ""
                            local[value.name] = dependencies.get(name, set())
                nested_dependencies = _graph_dependencies(nested, dependencies, local)
                sources |= set().union(
                    *(nested_dependencies.get(value.name, set()) for value in nested.output)
                )
        for name in node.output:
            dependencies[name] = set(sources)
    return dependencies


def _inspect_loops(graph, inherited, path="graph"):
    """Require each scan recurrence to use both image values and previous state."""
    dependencies = dict(inherited)
    for initializer in graph.initializer:
        dependencies[initializer.name] = set()
    results = []
    for index, node in enumerate(graph.node):
        node_path = f"{path}/{node.name or node.op_type + '_' + str(index)}"
        if node.op_type == "Loop":
            bodies = [attribute.g for attribute in node.attribute if attribute.type == 5]
            if len(bodies) != 1:
                raise ValueError(f"Loop has no unique body: {node_path}")
            body = bodies[0]
            local = {}
            for position, value in enumerate(body.input):
                local[value.name] = {f"__carried_state_{position}__"} if position >= 2 else set()
            body_dependencies = _graph_dependencies(body, dependencies, local)
            recurrence = []
            for position in range(2, len(body.input)):
                output_position = position - 1
                if output_position >= len(body.output):
                    raise ValueError(f"Invalid Loop state/output arity: {node_path}")
                sources = body_dependencies.get(body.output[output_position].name, set())
                if IMAGE_DEPENDENCY in sources and f"__carried_state_{position}__" in sources:
                    recurrence.append(body.input[position].name)
            if not recurrence:
                raise ValueError(
                    f"Scan Loop does not update a recurrent state from image values: {node_path}"
                )
            results.append({"node": node_path, "input_dependent_states": recurrence})
        for attribute in node.attribute:
            if attribute.type == 5:
                results.extend(_inspect_loops(attribute.g, dependencies, node_path))
        sources = set().union(*(dependencies.get(name, set()) for name in node.input))
        if node.op_type in {"Constant", "Shape", "Size"}:
            sources = set()
        for attribute in node.attribute:
            if attribute.type == 5:
                nested_dependencies = _graph_dependencies(attribute.g, dependencies)
                sources |= set().union(
                    *(nested_dependencies.get(value.name, set()) for value in attribute.g.output)
                )
        for name in node.output:
            dependencies[name] = set(sources)
    return results


def inspect_graph(path, checkpoint_sha256):
    """Reject legacy frozen scans and graphs lacking verifiable scan provenance."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    required = {
        "crackmamba.scan_backend": "onnx_loop",
        "crackmamba.scan_count": "8",
        "crackmamba.checkpoint_sha256": checkpoint_sha256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"ONNX metadata {key!r} must equal {expected!r}; re-export the model")
    graph = model.graph
    if [value.name for value in graph.input] != ["images"]:
        raise ValueError("Expected one ONNX input named 'images'")
    if [value.name for value in graph.output] != ["probability"]:
        raise ValueError("Expected one ONNX output named 'probability'")
    shape = [dimension.dim_value for dimension in graph.input[0].type.tensor_type.shape.dim]
    if shape != [1, 3, 480, 480]:
        raise ValueError(f"Expected fixed ONNX input [1, 3, 480, 480], received {shape}")
    if graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Expected float32 ONNX image input")
    output_shape = [dimension.dim_value for dimension in graph.output[0].type.tensor_type.shape.dim]
    if output_shape != [1, 1, 480, 480] or graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Expected float32 ONNX probability output [1, 1, 480, 480]")

    def reject_custom_ops(current):
        for initializer in current.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError("External-data ONNX models are not supported by this release gate")
        for node in current.node:
            if node.domain not in {"", "ai.onnx"}:
                raise ValueError(f"Unregistered custom ONNX operation: {node.domain}::{node.op_type}")
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    reject_custom_ops(attribute.g)

    reject_custom_ops(graph)
    loops = _inspect_loops(graph, {"images": {IMAGE_DEPENDENCY}})
    if len(loops) < 8:
        raise ValueError(f"Expected at least 8 input-dependent selective-scan Loops, received {len(loops)}")
    return {"status": "passed", "scan_loops": loops, "input_shape": shape, "metadata": metadata}


def compare_probabilities(reference, candidate, atol, rtol, max_mask_disagreement):
    """Check the full probability tensor and its thresholded segmentation mask."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[:2] != (1, 1):
        raise ValueError(f"Probability shape mismatch: {reference.shape} versus {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Probability tensor contains non-finite values")
    for name, value in (("reference", reference), ("candidate", candidate)):
        if value.min() < -1e-6 or value.max() > 1 + 1e-6:
            raise ValueError(f"{name} probability is outside [0, 1]")
    absolute_error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    mask_disagreement = float(np.mean((reference >= 0.5) != (candidate >= 0.5)))
    passed = bool(np.allclose(candidate, reference, atol=atol, rtol=rtol))
    passed = passed and mask_disagreement <= max_mask_disagreement
    return {
        "passed": passed,
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "mask_disagreement": mask_disagreement,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Real input image or directory of prepared RGB patches")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--engine", type=Path, help="Also compare an engine built on this Jetson")
    parser.add_argument("--trt-device", type=int, help="TensorRT GPU index (default: reference CUDA index, or 0 for a CPU reference)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scan-backend", choices=("mamba", "reference"), default="mamba")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Threads for the sequential CPU reference backend")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-mask-disagreement", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.max_samples < 1 or args.cpu_threads < 1 or not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0 or not 0 <= args.max_mask_disagreement <= 1:
        parser.error("Sample count must be positive and tolerances must be nonnegative (mask fraction <= 1)")
    if args.trt_device is not None and args.trt_device < 0:
        parser.error("TensorRT GPU index must be nonnegative")
    protected_inputs = [args.checkpoint, args.onnx, args.images]
    if args.engine:
        protected_inputs.append(args.engine)
    if any(args.report.resolve() == path.resolve() for path in protected_inputs):
        parser.error("The report path must differ from all model and input paths")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "checkpoint": {"path": str(args.checkpoint.resolve())},
        "onnx": {"path": str(args.onnx.resolve())},
        "reference_backend": args.scan_backend,
        "device": args.device,
        "tolerances": {"atol": args.atol, "rtol": args.rtol, "max_mask_disagreement": args.max_mask_disagreement},
        "numeric_validation": {"samples": 0, "results": []},
    }
    runner = None
    try:
        checkpoint_hash = sha256_file(args.checkpoint)
        report["checkpoint"]["sha256"] = checkpoint_hash
        report["onnx"]["sha256"] = sha256_file(args.onnx)
        report["graph_inspection"] = inspect_graph(args.onnx, checkpoint_hash)
        import onnxruntime as ort
        import torch
        from models.crackseu_global import load_model
        from preprocess import collect_images, load_image

        if args.device == "cpu":
            torch.set_num_threads(args.cpu_threads)

        image_paths = collect_images(args.images)[:args.max_samples]
        if not image_paths:
            raise ValueError("No input images found")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(str(args.onnx), session_options, providers=["CPUExecutionProvider"])
        model = load_model(args.checkpoint, args.device, scan_backend=args.scan_backend)
        model.eval()
        report["versions"] = {"torch": torch.__version__, "onnxruntime": ort.__version__}
        if args.engine:
            from trt_runtime import TensorRTStaticRunner
            device_index = args.trt_device
            if device_index is None:
                device_index = int(args.device.partition(":")[2] or 0) if args.device.startswith("cuda") else 0
            runner = TensorRTStaticRunner(args.engine, device=device_index)
            if set(runner.input_specs) != {"images"} or set(runner.output_specs) != {"probability"}:
                raise ValueError("Expected one TensorRT input 'images' and one output 'probability'")
            for name, spec, expected_shape in (
                ("images", runner.input_specs["images"], (1, 3, 480, 480)),
                ("probability", runner.output_specs["probability"], (1, 1, 480, 480)),
            ):
                if tuple(spec["shape"]) != expected_shape or spec["dtype"] != np.dtype("float32"):
                    raise ValueError(f"TensorRT {name} must have shape {expected_shape} and external float32 dtype")
            report["engine"] = {"path": str(args.engine.resolve()), "sha256": sha256_file(args.engine), "device_index": device_index}

        for image_path in image_paths:
            images = load_image(image_path)
            with torch.inference_mode():
                probability = model(torch.from_numpy(images).to(args.device)).detach().float().cpu().numpy()
            onnx_probability = session.run(["probability"], {"images": images})[0]
            result = {"image": str(image_path.resolve()), "image_sha256": sha256_file(image_path)}
            result["onnx"] = compare_probabilities(probability, onnx_probability, args.atol, args.rtol, args.max_mask_disagreement)
            if runner is not None:
                engine_probability = runner.infer({"images": images})["probability"]
                result["tensorrt"] = compare_probabilities(probability, engine_probability, args.atol, args.rtol, args.max_mask_disagreement)
            report["numeric_validation"]["results"].append(result)
            report["numeric_validation"]["samples"] += 1
            if not all(result[name]["passed"] for name in ("onnx", "tensorrt") if name in result):
                raise ValueError(f"Numerical validation failed for {image_path}; see the report for measured errors")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception as error:
                report["status"] = "failed"
                report["error"] = f"Failed to release TensorRT resources: {error}"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation {report['status']}: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
