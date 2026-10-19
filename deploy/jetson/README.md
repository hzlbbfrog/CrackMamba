# 🛫 Deploy CrackMamba on Jetson with TensorRT

This tutorial deploys **CrackSeU + CrackMamba_G pretrained on Deepcrack**. The bundled G-only model differs from the Steelcrack G+L training model in the repository root.

[Before you start](#-1-before-you-start) · [Deployment](#-2-deployment) · [Outputs](#-3-outputs)

## 🔧 1. Before you start

See [Environment](../../README.md#-1-environment) and [Install Mamba](../../README.md#-2-install-mamba) in the main README. TensorRT 10 and PyCUDA are assumed to be installed; Python dependencies are listed in [requirements.txt](requirements.txt).

Use **480 × 480 RGB images** with **batch size 1**. Three prepared [sample patches](samples/README.md) and the matching checkpoint are included.

Run all commands from the repository root in the same shell:

```bash
CHECKPOINT=deploy/jetson/checkpoints/crackmamba_global_deepcrack_epoch15.pth
IMAGES=deploy/jetson/samples
OUT=deploy/jetson/artifacts
```

> TensorRT compatibility, numerical agreement, and speed require validation on the target Orin.

## ⚡ 2. Deployment

![PyTorch to ONNX to TensorRT deployment workflow](../../figures/TensorRT.svg)

### 🐍 Step 1 — Run PyTorch inference

```bash
python deploy/jetson/infer_pytorch.py \
  --checkpoint "$CHECKPOINT" \
  --images "$IMAGES" \
  --output-dir "$OUT/pytorch" \
  --device cpu --scan-backend reference \
  --warmup 1 --threshold 0.5
```

This establishes the CPU reference. For optimized PyTorch inference and validation on Orin, use `--device cuda --scan-backend mamba` instead of the CPU settings.

### 📦 Step 2 — Export ONNX

```bash
python deploy/jetson/export_onnx.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT/crackmamba_global_480.onnx"
```

### 🔍 Step 3 — Validate ONNX

```bash
python deploy/jetson/validate_export.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$OUT/crackmamba_global_480.onnx" \
  --images "$IMAGES" \
  --device cpu --scan-backend reference \
  --report "$OUT/onnx_validation.json"
```

The validation report must pass before the engine can be built.

### 🏗️ Step 4 — Build the TensorRT engine

Run this step on the target Orin:

```bash
bash deploy/jetson/build_engine.sh \
  --checkpoint "$CHECKPOINT" \
  --onnx "$OUT/crackmamba_global_480.onnx" \
  --validation-report "$OUT/onnx_validation.json" \
  --engine "$OUT/crackmamba_global_480_fp32.plan"
```

Start with FP32. After validating it in Step 5, try FP16 by adding `--fp16`, choosing a separate engine filename, and validating that engine again. Export and engine building require fresh output filenames when rerun.

### ✅ Step 5 — Validate the TensorRT engine

```bash
python deploy/jetson/validate_export.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$OUT/crackmamba_global_480.onnx" \
  --engine "$OUT/crackmamba_global_480_fp32.plan" \
  --images "$IMAGES" \
  --device cpu --scan-backend reference \
  --report "$OUT/tensorrt_validation.json"
```

### 🚀 Step 6 — Run TensorRT inference

```bash
python deploy/jetson/infer_tensorrt.py \
  --engine "$OUT/crackmamba_global_480_fp32.plan" \
  --images "$IMAGES" \
  --output-dir "$OUT/tensorrt" \
  --threshold 0.5
```

To use your own images, set `IMAGES` to a directory of prepared 480 × 480 RGB patches. Keep the output directory outside the input directory.

## 📁 3. Outputs

The ONNX model, TensorRT engine, and validation reports are written under `$OUT`. Both inference scripts save predictions in their specified output directories:

| File | Contents |
| --- | --- |
| `sample_01.png.probability.npy` | Float32 foreground probabilities |
| `sample_01.png.probability.png` | Grayscale probability preview |
| `sample_01.png.binary.png` | Binary mask at threshold 0.5 |
| `report.json` | Inference timing and configuration |

Timing includes data transfers and model execution; it excludes image loading, preprocessing, postprocessing, and file saving.
