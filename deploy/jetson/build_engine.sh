#!/usr/bin/env bash
# Build on the target Jetson only after graph and numerical validation pass.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash build_engine.sh --onnx MODEL.onnx --checkpoint MODEL.pth \
    --validation-report validation.json --engine MODEL.plan [options]

Options:
    --fp16                   Enable FP16 tactics (external I/O remains graph-defined)
    --workspace-mib N        Workspace limit in MiB (default: 2048)
    --trtexec PATH           TensorRT 10 trtexec binary
    --python PATH            Python interpreter (default: python3)
    --log PATH               Build log (default: ENGINE.build.log)

The output engine must not already exist. Validate the newly built engine with
validate_export.py --engine before using it for predictions or benchmarks.
USAGE
}

onnx_path=""
checkpoint_path=""
validation_report=""
engine_path=""
log_path=""
trtexec_path=""
python_path="python3"
workspace_mib="2048"
fp16=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx|--checkpoint|--validation-report|--engine|--log|--trtexec|--python|--workspace-mib)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --onnx) onnx_path="$2" ;;
                --checkpoint) checkpoint_path="$2" ;;
                --validation-report) validation_report="$2" ;;
                --engine) engine_path="$2" ;;
                --log) log_path="$2" ;;
                --trtexec) trtexec_path="$2" ;;
                --python) python_path="$2" ;;
                --workspace-mib) workspace_mib="$2" ;;
            esac
            shift 2
            ;;
        --fp16) fp16=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$onnx_path" && -n "$checkpoint_path" && -n "$validation_report" && -n "$engine_path" ]] || { usage >&2; exit 2; }
[[ "$workspace_mib" =~ ^[1-9][0-9]*$ ]] || { echo "Workspace must be a positive integer" >&2; exit 2; }
[[ ! -e "$engine_path" ]] || { echo "Engine already exists; choose a new output path: $engine_path" >&2; exit 2; }
[[ -n "$log_path" ]] || log_path="${engine_path}.build.log"
if [[ -z "$trtexec_path" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        trtexec_path="$(command -v trtexec)"
    else
        trtexec_path="/usr/src/tensorrt/bin/trtexec"
    fi
fi
[[ -x "$trtexec_path" ]] || { echo "TensorRT trtexec not found: $trtexec_path" >&2; exit 2; }

"$python_path" - "$onnx_path" "$checkpoint_path" "$validation_report" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

onnx_path, checkpoint_path, report_path = map(Path, sys.argv[1:])
try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise ValueError("The validation report must have status='passed' and schema_version=1")
    if report.get("graph_inspection", {}).get("status") != "passed":
        raise ValueError("The report must include a successful scan graph inspection")
    if len(report["graph_inspection"].get("scan_loops", [])) < 8:
        raise ValueError("The report must cover all eight selective scans")
    numeric = report.get("numeric_validation", {})
    results = numeric.get("results", [])
    if numeric.get("samples", 0) < 1 or len(results) != numeric["samples"]:
        raise ValueError("The report must contain at least one executed numerical comparison")
    if any(not result.get("onnx", {}).get("passed", False) or not result.get("image_sha256") for result in results):
        raise ValueError("Every ONNX comparison must pass on a recorded real input image")
    for name, path in (("onnx", onnx_path), ("checkpoint", checkpoint_path)):
        if digest(path) != report.get(name, {}).get("sha256"):
            raise ValueError(f"{name} SHA256 does not match the validation report; validate again")
except (OSError, ValueError, KeyError, TypeError) as error:
    raise SystemExit(f"Build gate rejected the artifacts: {error}")
print("Validation gate passed: graph, checkpoint, and numerical report match.")
PY

version_output="$("$trtexec_path" --version 2>&1)"
"$python_path" - "$version_output" <<'PY'
import re
import sys
match = re.search(r"TensorRT[^\n]*?\bv?(\d+(?:\.\d+){0,3})", sys.argv[1], re.IGNORECASE)
if match is None:
    raise SystemExit("Could not identify the TensorRT version from trtexec --version")
version = match.group(1)
major = int(version.split(".")[0])
if "." not in version and major >= 100000:
    major //= 10000
if major != 10:
    raise SystemExit(f"TensorRT 10 is required by this build workflow; received {version}")
print(f"Building with TensorRT {version}")
PY

mkdir -p -- "$(dirname -- "$engine_path")" "$(dirname -- "$log_path")"
command=("$trtexec_path" "--onnx=$onnx_path" "--saveEngine=$engine_path" "--memPoolSize=workspace:$workspace_mib" "--noTF32" "--skipInference")
if [[ "$fp16" -eq 1 ]]; then
    command+=(--fp16)
fi
printf '%s\n' "$version_output" > "$log_path"
printf 'Command:' >> "$log_path"
printf ' %q' "${command[@]}" >> "$log_path"
printf '\n' >> "$log_path"
"${command[@]}" 2>&1 | tee -a "$log_path"
[[ -s "$engine_path" ]] || { echo "TensorRT did not produce a nonempty engine" >&2; exit 1; }

"$python_path" - "$onnx_path" "$checkpoint_path" "$validation_report" "$engine_path" "$log_path" "$fp16" "$version_output" "${command[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

onnx_path, checkpoint_path, report_path, engine_path, log_path, fp16, version = sys.argv[1:8]
manifest = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "fp16_tactics_enabled": fp16 == "1",
    "trtexec_version": version,
    "command": sys.argv[8:],
    "build_log": str(Path(log_path).resolve()),
    "engine_numerically_validated": False,
}
for name, path in (("onnx", onnx_path), ("checkpoint", checkpoint_path), ("validation_report", report_path), ("engine", engine_path)):
    manifest[name] = {"path": str(Path(path).resolve()), "sha256": digest(path)}
manifest_path = Path(engine_path + ".build.json")
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Saved engine build manifest: {manifest_path}")
PY
echo "Engine built. Run validate_export.py --engine before inference or benchmarking."
