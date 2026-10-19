"""Record the installed Jetson inference environment without changing packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").replace("\x00", "").strip()
    except (OSError, UnicodeError):
        return None


def package_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def command_output(command):
    if not shutil.which(command[0]) and not Path(command[0]).is_file():
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def collect_environment():
    report = {
        "schema_version": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
        "jetson": {"device_model": read_text("/proc/device-tree/model"), "l4t_release": read_text("/etc/nv_tegra_release"), "os_release": read_text("/etc/os-release")},
        "packages": {name: package_version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "onnx", "onnxruntime", "mamba-ssm", "causal-conv1d", "pycuda")},
        "cuda": {"available": False},
        "tensorrt": {"version": package_version("tensorrt", "tensorrt-lean", "tensorrt-cu12")},
        "commands": {},
    }
    try:
        import torch
        report["packages"]["torch"] = torch.__version__
        report["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version()}
        if torch.cuda.is_available():
            report["cuda"]["devices"] = [{"index": index, "name": torch.cuda.get_device_name(index), "capability": list(torch.cuda.get_device_capability(index))} for index in range(torch.cuda.device_count())]
    except Exception as error:
        report["cuda"]["error"] = f"{type(error).__name__}: {error}"
    try:
        import tensorrt as trt
        report["tensorrt"]["version"] = trt.__version__
        report["tensorrt"]["importable"] = True
    except Exception as error:
        report["tensorrt"]["importable"] = False
        report["tensorrt"]["error"] = f"{type(error).__name__}: {error}"
    commands = {
        "cuda_compiler": ["nvcc", "--version"],
        "jetpack_packages": ["dpkg-query", "-W", "nvidia-jetpack", "nvidia-l4t-core", "libnvinfer10"],
        "power_mode": ["nvpmodel", "-q"],
        "clocks": ["jetson_clocks", "--show"],
    }
    for label, command in commands.items():
        report["commands"][label] = command_output(command)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the environment record to this JSON file")
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless PyTorch can use CUDA")
    parser.add_argument("--require-tensorrt", action="store_true", help="Fail unless TensorRT 10 can be imported")
    args = parser.parse_args(argv)
    report = collect_environment()
    valid = not args.require_cuda or report["cuda"]["available"]
    if args.require_tensorrt:
        version = report["tensorrt"]["version"] or "0"
        major = re.match(r"(\d+)", version)
        valid = valid and report["tensorrt"]["importable"] and major is not None and int(major.group(1)) == 10
    report["requirements_satisfied"] = bool(valid)
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
