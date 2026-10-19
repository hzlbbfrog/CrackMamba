"""Run a static G-only TensorRT engine and save its foreground probabilities."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-name", default="probability")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--resize", action="store_true", help="Explicitly bilinear-resize inputs")
    parser.add_argument("--threshold", type=float, help="Also save binary masks at this threshold")
    parser.add_argument("--report", type=Path, help="JSON report; defaults to OUTPUT_DIR/report.json")
    return parser


def main(argv=None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    import numpy as np
    from preprocess import (collect_images, load_image, save_prediction, sha256_file,
                            validate_output_directory, write_report)
    from trt_runtime import TensorRTRunner

    validate_output_directory(args.images, args.output_dir)
    images = collect_images(args.images)
    durations = []
    results = []
    with TensorRTRunner(args.engine, device=args.device) as runner:
        if len(runner.input_specs) != 1:
            raise ValueError("This CLI expects a single RGB image input")
        input_name, spec = next(iter(runner.input_specs.items()))
        shape = spec["shape"]
        if len(shape) != 4 or shape[:2] != (1, 3):
            raise ValueError(f"Expected a batch-one RGB NCHW input; engine has {shape}")
        if spec["dtype"] not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError(f"Expected a floating-point image input, got {spec['dtype']}")
        if args.output_name not in runner.output_specs:
            raise ValueError(f"Unknown output {args.output_name!r}; choose from {list(runner.output_specs)}")
        output_shape = runner.output_specs[args.output_name]["shape"]
        if output_shape != (1, 1, shape[2], shape[3]):
            raise ValueError(f"Expected a full-resolution probability output, got {output_shape}")
        size = (shape[2], shape[3])

        def prepare(image):
            return load_image(image, size=size, resize=args.resize).astype(spec["dtype"], copy=False)

        first = prepare(images[0])
        for _ in range(args.warmup):
            runner.infer({input_name: first})
        for image in images:
            array = prepare(image)
            started = time.perf_counter()
            probability = runner.infer({input_name: array})[args.output_name]
            durations.append(time.perf_counter() - started)
            outputs = save_prediction(probability, image, args.images, args.output_dir, args.threshold)
            results.append({"image": str(image), "outputs": outputs})
        elapsed = sum(durations)
        report = {
            "backend": "tensorrt", "engine": str(args.engine.resolve()),
            "engine_sha256": sha256_file(args.engine), "device": args.device,
            "device_name": runner.device_name, "tensorrt": runner.tensorrt_version,
            "python": platform.python_version(), "numpy": np.__version__,
            "input_name": input_name, "input_shape": list(shape),
            "input_dtype": str(spec["dtype"]), "output_name": args.output_name,
            "output_shape": list(output_shape),
            "output_dtype": str(runner.output_specs[args.output_name]["dtype"]),
            "normalization": "RGB; (pixel / 255 - 0.5) / 0.5", "resize": args.resize,
            "warmup_iterations": args.warmup, "images": len(images), "batch_size": 1,
            "timing_scope": "Per-image host-buffer copy + synchronized H2D + execute + D2H + "
                            "output-array copy; excludes image decode, preprocessing, prediction "
                            "validation, output saving and report writing",
            "total_inference_seconds": elapsed,
            "mean_latency_ms": elapsed / len(images) * 1000.0,
            "median_latency_ms": float(np.median(durations)) * 1000.0,
            "throughput_images_per_second": len(images) / elapsed,
            "threshold": args.threshold, "predictions": results,
        }
    destination = args.report or args.output_dir / "report.json"
    write_report(report, destination)
    print(f"Saved {len(images)} predictions and report: {destination}")
    print(f"Mean inference latency: {report['mean_latency_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
