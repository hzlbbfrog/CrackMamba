"""Run the Deepcrack G-only checkpoint as a probability-preserving reference."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU inference thread count")
    parser.add_argument("--scan-backend", choices=("mamba", "reference"), default="mamba")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--size", type=int, nargs=2, default=(480, 480), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--resize", action="store_true", help="Explicitly bilinear-resize inputs")
    parser.add_argument("--threshold", type=float, help="Also save binary masks at this threshold")
    parser.add_argument("--report", type=Path, help="JSON report; defaults to OUTPUT_DIR/report.json")
    return parser


def main(argv=None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    import numpy as np
    import torch
    from models.crackseu_global import load_model
    from preprocess import (collect_images, load_image, save_prediction, sha256_file,
                            validate_output_directory, write_report)

    validate_output_directory(args.images, args.output_dir)
    images = collect_images(args.images)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model = load_model(args.checkpoint, device=device, scan_backend=args.scan_backend)
    model.eval()
    size = tuple(args.size)
    first = load_image(images[0], size=size, resize=args.resize)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def predict(array):
        tensor = torch.from_numpy(array).to(device)
        output = model(tensor)
        if not isinstance(output, torch.Tensor):
            raise TypeError("The deployment model must return a single probability tensor")
        return output.detach().to(device="cpu", dtype=torch.float32).numpy().copy()

    results = []
    durations = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            predict(first)
        synchronize()
        for image in images:
            array = load_image(image, size=size, resize=args.resize)
            synchronize()
            started = time.perf_counter()
            probability = predict(array)
            synchronize()
            durations.append(time.perf_counter() - started)
            outputs = save_prediction(probability, image, args.images, args.output_dir, args.threshold)
            results.append({"image": str(image), "outputs": outputs})

    elapsed = sum(durations)
    report = {
        "backend": "pytorch", "scan_backend": args.scan_backend,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device), "python": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "pytorch": torch.__version__, "numpy": np.__version__,
        "input_shape": list(first.shape), "input_dtype": "float32",
        "normalization": "RGB; (pixel / 255 - 0.5) / 0.5", "resize": args.resize,
        "warmup_iterations": args.warmup, "images": len(images), "batch_size": 1,
        "timing_scope": "Per-image synchronized H2D + forward + D2H; excludes image decode, "
                        "preprocessing, prediction validation, output saving and report writing",
        "total_inference_seconds": elapsed,
        "mean_latency_ms": elapsed / len(images) * 1000.0,
        "median_latency_ms": float(np.median(durations)) * 1000.0,
        "throughput_images_per_second": len(images) / elapsed,
        "threshold": args.threshold, "predictions": results,
    }
    if device.type == "cuda":
        report["device_name"] = torch.cuda.get_device_name(device)
        report["cuda_runtime"] = torch.version.cuda
    destination = args.report or args.output_dir / "report.json"
    write_report(report, destination)
    print(f"Saved {len(images)} predictions and report: {destination}")
    print(f"Mean inference latency: {report['mean_latency_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
