# Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **Notice:** This repository contains the official implementation of the manuscript **"Efficient 3D Hand Mesh Reconstruction: Lightweight Anchor-Guided Multi-Scale Interaction"**, which has been submitted to **The Visual Computer**.
>

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
```

## 🔗 Dataset
Please download the datasets and organize the directory as follows:

```text
data
├── Compdata
│   ├── base_pose
│   ├── trans_pose_batch1
│   ├── trans_pose_batch2
│   └── trans_pose_batch3
├── FreiHAND
│   ├── evaluation
│   ├── training
│   ├── cmr_g.json
│   ├── evaluation_*.json
│   ├── training_*.json
│   └── information.txt
└── Ge
    ├── images
    ├── params.mat
    └── pose_gt.mat

###  FreiHAND

*Please download FreiHAND dataset from [this link](https://lmb.informatik.uni-freiburg.de/projects/freihand/), and create a soft link in `data`, i.e., `data/FreiHAND`.
*   Download mesh GT file `freihand_train_mesh.zip`, and unzip it under `data/FreiHAND/training`.

### Real World Testset

*   Please download the dataset from [this link](https://github.com/3d-hand-shape/hand-graph-cnn/tree/master/data/real_world_testset), and create a soft link in `data`, i.e., `data/Ge`.

### Complement data

*   See [this file](https://github.com/SeanChenxy/HandMesh/blob/main/complement_data.md)for complement data. Then, create a soft link in `data`, i.e., `data/Compdata`.
```

## train
```bash
python main.py --exp_name Lite_AMNet --PHASE train --Local_testing --config_file .\configs\Lite-AMNet.yml
```

## 🔗 Evaluation
To evaluate our model, please download the pre-trained weights via [this link](https://drive.google.com/drive/folders/1Ij-vkdMv3nXI4xbRcT8X6isyaWCiRJkU?usp=drive_link) and place them in the `checkpoints` folder.
```bash
python tools/freihand-master/eval.py /home/hdh/桌面/zjj_Datas/Lite-AMNet/data/FreiHAND /home/hdh/桌面/zjj_Datas/Lite-AMNet/out/MultipleDatasets/Lite_AMNet --pred_file_name /home/hdh/桌面/zjj_Datas/Lite-AMNet/out/MultipleDatasets/Lite_AMNet/Lite_AMNet0.json

```

## 🔗 Visualize model
```bash
python main.py --PHASE demo_test_new_data --exp_name mobrecon_spconv --config_file ./configs/Lite-AMNet.yml --opts MODEL.NAME LiteSpiralGCN TRAIN.GPU_ID "[-1]" DATA.FREIHAND.USE True TRAIN.DATASET FreiHAND VAL.DATASET FreiHAND TEST.DATASET FreiHAND MODEL.RESUME "checkpoints path"
```
## Acknowledgement
We acknowledge the open-source contributions from the following repositories.

[HandMesh](https://github.com/SeanChenxy/HandMesh.git)

[hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn.git)

[I2L-MeshNet_RELEASE](https://github.com/mks0601/I2L-MeshNet_RELEASE.git)

[detectron2](https://github.com/facebookresearch/detectron2.git)

[freihand](https://github.com/lmb-freiburg/freihand.git)

