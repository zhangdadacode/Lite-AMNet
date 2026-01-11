# Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **Notice:** This repository contains the official implementation of the manuscript **"Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction"**, which has been submitted to **The Visual Computer**.
>
> If you find this code or our method useful for your research, please consider citing our paper (see [Citation](#citation) below).

## 📝 Abstract

<!-- 提示：这里直接粘贴编辑帮您修改好的那段Abstract，完全一致能增加好感度 -->
In the realm of Virtual Reality (VR) and Augmented Reality (AR), accurate 3D hand modelling is pivotal. This paper introduces Lite-AMNet, a lightweight framework for 3D hand mesh reconstruction from a single RGB image. Addressing the challenge of balancing reconstruction accuracy and computational efficiency, Lite-AMNet employs an anchor-assisted attention sampling mechanism and a Shuffle Attention Combination Network (SACN) to enhance feature representation. Additionally, a Shared Compressed Attention Fusion (SCAF) module and a Multi-Scale Depthwise Separable Convolution (MS-DSConv) module are introduced to improve multi-scale feature interaction and geometric representation. Experiments on the FreiHAND dataset demonstrate superior performance, achieving a PA-MPJPE of 6.5 mm and a PA-MPVPE of 6.6 mm with only 9.47 million parameters, while maintaining a real-time inference speed of 73 FPS. These results underscore Lite-AMNet's potential for real-time applications in resource-constrained environments.

## 🖼️ Visual Results

<!-- 建议：放一张 Teaser 图，展示输入图片和重建后的 3D 网格对比，或者 GIF 动图 -->
<div align="center">
  <img src="assets/teaser.png" width="800px" />
  <p>Figure 1: Visual comparison of our method on the FreiHAND dataset.</p>
</div>

## 🛠️ Installation

Please follow the steps below to set up the environment.

```bash
# 1. Clone the repository
git clone https://github.com/[YOUR_USERNAME]/[YOUR_REPO_NAME].git
cd [YOUR_REPO_NAME]

# 2. Create a conda environment
conda create -n liteamnet python=3.8
conda activate liteamnet

# 3. Install dependencies
# It is recommended to install PyTorch matching your CUDA version first
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# 4. Install Manopth (for MANO layer)
# (根据您的具体情况，如果用了 manopth 或 smplx，请保留此步)
pip install manopth
