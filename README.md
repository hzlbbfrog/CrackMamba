<div align="center">
  <img src="./figures/Logo.svg" width="100%">
</div>

# CrackMamba
This repository is the official PyTorch implementation of Mamba meets crack segmentation.

### 🍓 Paper:
- The initial version of the paper can refer to the [arXiv version](https://arxiv.org/abs/2407.15714) or [ResearchGate](https://www.researchgate.net/publication/382459254_Mamba_meets_crack_segmentation).  

## 📢 Introduction
This repository mainly includes 2 parts:  
- [ ] Code of **CrackMamba**
- [ ] A simple tutorial about how to install Mamba. → [here](https://github.com/hzlbbfrog/CrackMamba?tab=readme-ov-file#2-installation-of-mamba)
- [ ] A simple tutorial about how to play deep learning on a Jetson device. → [here](https://github.com/hzlbbfrog/CrackMamba?tab=readme-ov-file#-build-a-deep-learning-environment-on-a-jetson-device)

## 📌 Updates
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

**Step 3:** Install Mamba 1.2.0

## 🛫 Build a deep learning environment on a Jetson device


## 💗 Cite CrackMamba

## 💘 Acknowledgements


