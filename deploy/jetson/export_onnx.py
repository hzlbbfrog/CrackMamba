"""Export the Deepcrack G-only network with eight explicit ONNX scan Loops."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if (args.height, args.width) != (480, 480):
        parser.error("This deployment export and release validation use fixed 480x480 input")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve validated artifacts")
    return args


def main(argv=None):
    args = parse_args(argv)
    import onnx
    import torch
    from models.crackseu_global import load_model
    from models.onnx_scan import lower_selective_scans

    torch.set_num_threads(args.cpu_threads)
    model = load_model(args.checkpoint, args.device, scan_backend="reference")
    example = torch.zeros(1, 3, args.height, args.width, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_suffix(args.output.suffix + ".bridge.tmp")
    if intermediate.exists():
        raise FileExistsError(f"Stale export intermediate exists: {intermediate}")
    try:
        with torch.inference_mode(), warnings.catch_warnings():
            # The bridge supplies real NumPy tracing values; its explicit symbolic
            # carries all scan inputs into ONNX and is lowered below, never frozen.
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(model, example, str(intermediate), export_params=True,
                opset_version=19, input_names=["images"], output_names=["probability"],
                custom_opsets={"crackmamba": 1}, dynamo=False, do_constant_folding=True)
        graph = lower_selective_scans(onnx.load(str(intermediate)))
        metadata = {
            "crackmamba.scan_backend": "onnx_loop",
            "crackmamba.scan_count": "8",
            "crackmamba.checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "crackmamba.input_shape": json.dumps([1, 3, args.height, args.width]),
            "crackmamba.output": "probability",
            "crackmamba.architecture": "CrackSeU_Global_Deepcrack",
            "crackmamba.tensorrt_status": "requires_target_device_validation",
        }
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(args.output))
    finally:
        intermediate.unlink(missing_ok=True)
    print(f"Exported {args.output} with eight input-dependent ONNX scan Loops.")
    print("Run validate_export.py before engine construction. Orin performance is not established by this export.")


if __name__ == "__main__":
    main()
