<div align="center">
  <img src="./figures/Logo.svg" width="100%">
</div>

# CrackMamba
This repository is the official PyTorch implementation of **[CrackMamba](https://www.sciencedirect.com/science/article/pii/S0957417426019469)**.

### 🍓 Paper:
- The initial version of the paper can refer to the [arXiv version](https://arxiv.org/abs/2407.15714) or [ResearchGate](https://www.researchgate.net/publication/382459254_Mamba_meets_crack_segmentation).
- The final journal version of our paper can refer to the [journal version](https://www.sciencedirect.com/science/article/pii/S0957417426019469).

## 📢 Introduction
This repository mainly includes 2 parts:  
- [ ] Code of **CrackMamba**
- [ ] A simple tutorial about how to install Mamba on PC. → [here](https://github.com/hzlbbfrog/CrackMamba?tab=readme-ov-file#2-installation-of-mamba)
- [ ] A simple tutorial about how to play deep learning on a Jetson device. → [here](https://github.com/hzlbbfrog/CrackMamba?tab=readme-ov-file#-build-a-deep-learning-environment-on-a-jetson-device)

## 📌 Updates
- **`2026/10/11`**: We release all the code.
- **`2026/05/26`**: Our paper is finally accepted by Expert Systems with Applications **(IF 9.4, CCF-C)** after **2 years** later!
- **`2024/07/23`**: The preprint of our paper is publicly available on arXiv. Link → [Arxiv Paper](https://arxiv.org/abs/2407.15714).
- **`2024/07/22`**: The preprint of our paper is submitted to arXiv.
- **`2024/06/30`**: This repository is built up! The code will be gradually released.

## 🏎️ Getting Started

### 1. Requirements
Our version requirements are **lower than** those of other Mamba-based research, which is **more friendly** to those who installed old versions.
~~~
Recommended versions are
    * Mamba = 1.1.1 or 1.2.0
    * causal-conv1d = 1.1.1
    * python = 3.8.15
    * pytorch = 1.12.1
    * CUDA 11.6.2 and CUDNN 8.6.0  
~~~
I want to let you know that the following installation **works in Linux (specifically, I used Ubuntu 18.04)**. It **does not work** when I try to reproduce it in Win 11. 😤

### 2. Installation of Mamba

**Step 1: Install causal_conv 1.1.1**
~~~
pip install causal_conv1d==1.1.1
~~~
The screenshot of the successful installation is shown below.
<div align="center">
  <img src="./figures/causal_conv1d 1.1.1.png" width="100%">
</div>

**Step 2: Install Mamba 1.2.0**
~~~
cd mamba-1.2.0
python setup.py install
~~~
Remark: We have provided the needed [Mamba-1.2.0](https://github.com/hzlbbfrog/CrackMamba/tree/main/mamba-1.2.0) in this repo for your convenience.  
The screenshot of the successful installation is shown below.
<div align="center">
  <img src="./figures/Mamba 1.2.0.png" width="100%">
</div>

### 3. Prepare the dataset

### 4. Training and validation

### 5. Test


## 🛫 Build a deep learning environment on a Jetson device
### 1. Basic environment

### 2. Fast inference based on TensorRT (V10)
In version 10+ of TensorRT, you should use **context.execute_async_v3()** for inference. **context.execute_async_v2()** has been deprecated.



## 💗 Cite CrackMamba
If you have any problems, please do not hesitate to contact us!
You are very welcome to cite our paper!  
The BibTeX entry of the paper is as follows:
```BibTeX
@article{CrackMamba,
title = {CrackMamba: An attention-guided Mamba architecture for efficient and edge-ready crack segmentation using global and local information},
journal = {Expert Systems with Applications},
volume = {329},
pages = {133035},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.133035},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426019469},
author = {Zhili He and Wai Yi Chau and Mei Ling Leung and Yingyue Han and Wessam Hamid and Yu-Hsing Wang and Jian Zhang},
keywords = {Deep learning, Computer vision, Crack segmentation, Mamba, Attentionmechanism}
}
```

## 💘 Acknowledgements
This repo benefits from [Vim](https://github.com/hustvl/Vim) and [VMamba](https://github.com/mzeromiko/vmamba).  
Thanks for their wonderful works!

