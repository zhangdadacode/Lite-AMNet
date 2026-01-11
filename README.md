# Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **Notice:** This repository contains the official implementation of the manuscript **"Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction"**, which has been submitted to **The Visual Computer**.
>

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

```bash
# 1. Clone the repository
git clone https://github.com/zhangdadacode/Lite-AMNet.git
cd Lite-AMNet

# 2. Create a conda environment
conda create -n liteamnet python=3.8
conda activate liteamnet

# 3. Install PyTorch matching your CUDA version
# For CUDA 11.3 (as used in our experiments):
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# For other CUDA versions, please refer to the official PyTorch website: https://pytorch.org/get-started/previous-versions/

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install Manopth (if needed)
# pip install manopth
