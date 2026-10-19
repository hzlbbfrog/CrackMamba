# CPU conversion validation

[`cpu_reference.json`](cpu_reference.json) records the local conversion check
performed on 2026-10-20 using the bundled Deepcrack checkpoint and all three
published sample patches.

| Check | Result |
| --- | --- |
| Strict checkpoint loading | All 237 entries matched |
| Reference backend | Float32 PyTorch recurrence, CPU |
| Export backend | Eight standard ONNX Loop recurrences |
| PyTorch / ONNX / ONNX Runtime | 2.6.0+cpu / 1.17.0 / 1.20.1 |
| NumPy | 1.26.4 |
| Prepared samples | Three RGB patches, each 480 x 480 |
| Maximum absolute probability error | 1.5437602996826172e-5 |
| Maximum mean absolute probability error | 7.4022e-8 |
| Binary-mask disagreement at 0.5 | Zero on each sample |
| Automated regression tests | 18 passed |

All eight scan states were confirmed to depend on image values and their previous
state. The old frozen-scan graph is rejected by the new validation checks.

The JSON uses repository-relative display paths for portability. Its artifact
hashes, numerical comparisons, and timestamps are retained from the actual run.
Generated ONNX files are rebuilt with the documented export command; validate
each newly generated graph and use its new report for engine construction.

These are CPU conversion and runtime-API regression checks. CUDA Mamba agreement,
TensorRT parser support, engine construction, GPU numerical agreement and Jetson
throughput remain to be checked on the target Orin. The sample patches have no
ground-truth masks, so these comparisons do not measure segmentation accuracy.
