# Deepcrack global-only checkpoint

`crackmamba_global_deepcrack_epoch15.pth` is the archived Deepcrack-trained
CrackSeU + CrackMamba_G checkpoint selected by the Orin export scripts.
It contains 237 state-dictionary entries, including the auxiliary-head parameters
and fixed spatial scan buffers. The inference model loads every entry strictly
and returns only the main probability map; the auxiliary head is not evaluated.

- Original architecture identifier: `CrackSeU_B_LN_VT_CrackMamba_2025_8path_2SSD_1d_HW`
- Training run identifier: `1_0`
- Training batch-size label: `12`
- Saved epoch: `15`
- Deployment batch size: `1`
- SHA256: `24c12cc000cce3a62c8027bb91ede3b97ba1b1362f432fe5bfbbc0283e6b9220`

The published file is byte-identical to the archived checkpoint; only its filename
has been shortened. CPU reference and ONNX agreement are recorded in
[`../validation/cpu_reference.json`](../validation/cpu_reference.json).
The checkpoint is not interchangeable with the root Steelcrack G+L model.
The recorded CPU checks do not establish Jetson TensorRT throughput or CUDA
Mamba equivalence. See the deployment README for the target-device checks.
