# 🖼️ Pavement inference samples

These three RGB patches are supplied with the CrackMamba field-deployment example. Each image is **480 × 480** pixels. They were copied without resizing or recompression from the authors' HKUST pavement image collection.

| File | Original patch | SHA256 |
| --- | --- | --- |
| `sample_01.png` | `images_1/000001.png` | `3723d74be269b28ecca5e56cb5716855f169f8c7fba4f42fda86b08e565aeaa9` |
| `sample_02.png` | `images_1/000090.png` | `8a27f028bb1d22c4781a9b15737988a1f147aa5589cbdbb0b56e2894741e0fb1` |
| `sample_03.png` | `images_1/000180.png` | `466fd6e6d3187b969edaa2757a559a12e70b38114d4a27cdfc6285349e7966bf` |

The samples let you check preprocessing, inference, export equivalence, and output saving. They do not include ground-truth masks and cannot establish segmentation accuracy or reproduce the paper's full benchmark. Use a representative evaluation set when validating a deployment.

The field workflow in Section 5 of [the paper](https://doi.org/10.1016/j.eswa.2026.133035) starts from 5120 × 5120 camera images, divides them into patches, and resizes the patches to 480 × 480. Camera capture and full-image tiling are outside this example. See the [deployment guide](../README.md) for the model, preprocessing, and commands.
