from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scipy.io as sio
import os.path as osp
import cv2
import numpy as np
import torch
import torch.utils.data
from utils.vis import base_transform, inv_base_tranmsform, uv2map
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from termcolor import cprint
from build import DATA_REGISTRY
from tools.vis import perspective


@DATA_REGISTRY.register()
class Ge(torch.utils.data.Dataset):
    def __init__(self, cfg, phase='eval', writer=None):
        self.cfg = cfg
        self.phase = phase
        self.mean = torch.tensor([0.0016, 0.0025, 0.7360]).float()
        self.std = torch.tensor(0.20)
        self.img_std = self.cfg.DATA.IMG_STD
        self.img_mean = self.cfg.DATA.IMG_MEAN
        self.root = self.cfg.DATA.GE.ROOT
        self.size = self.cfg.DATA.SIZE

        mat_params = sio.loadmat(os.path.join(self.root, 'params.mat'))
        self.image_paths = mat_params["image_path"]

        # N x 4, [fx, fy, u0, v0]
        self.cam_params = torch.from_numpy(mat_params["cam_param"]).float()
        assert len(self.image_paths) == self.cam_params.shape[0]

        # N x 4, bounding box in the original image, [x, y, w, h]
        self.bboxes = torch.from_numpy(mat_params["bbox"]).float()
        assert len(self.image_paths) == self.bboxes.shape[0]

        # N x 3, [root_x, root_y, root_z]
        self.pose_roots = torch.from_numpy(mat_params["pose_root"]).float()
        assert len(self.image_paths) == self.pose_roots.shape[0]

        if "pose_scale" in mat_params.keys():
            # N, length of the first bone of the middle finger
            self.pose_scales = torch.from_numpy(mat_params["pose_scale"]).squeeze().float()
        else:
            self.pose_scales = torch.ones(len(self.image_paths)) * 5.0
        assert len(self.image_paths) == self.pose_scales.shape[0]

        mat_gt = sio.loadmat(os.path.join(self.root, 'pose_gt.mat'))
        # N x K x 3
        self.pose_gts = torch.from_numpy(mat_gt["pose_gt"])
        assert len(self.image_paths) == self.pose_gts.shape[0]

        if writer is not None:
            writer.print_str('Loaded Ge test {} samples'.format(len(self.image_paths)))
        cprint('Loaded Ge test {} samples'.format(len(self.image_paths)), 'red')

    def __getitem__(self, idx):
        # --- Load raw data ---
        img = cv2.imread(osp.join(self.root, self.image_paths[idx]))[:, ::-1, ::-1]
        img = base_transform(img, self.size, std=self.img_std, mean=self.img_mean)
        bbox = self.bboxes[idx].clone()
        bbox[0] = 1280 - bbox[0] - bbox[2]
        xyz = self.pose_gts[idx].clone() / 100
        xyz[:, 0] *= -1

        xyz_root = self.pose_roots[idx].clone().unsqueeze(0) / 100
        xyz_root[:, 0] *= -1

        fx, fy, u0, v0 = self.cam_params[idx].clone()
        u0 = 1280 - u0
        scale = self.size / bbox[2]
        calib = np.eye(4)
        calib[0, 0] = fx * scale
        calib[1, 1] = fy * scale
        calib[0, 2] = scale * (u0 - bbox[0] + 0.5) - 0.5
        calib[1, 2] = scale * (v0 - bbox[1] + 0.5) - 0.5
        calib = torch.from_numpy(calib).float()
        uv = perspective(xyz.clone().T.unsqueeze(0), calib.unsqueeze(0))[0].numpy().T[:, :2]

        uv = uv / img.shape[1:][::-1]
        xyz -= xyz_root

        # Ensure contiguous, resizable tensors for DataLoader workers
        img = torch.from_numpy(np.ascontiguousarray(img)).float().clone()
        uv_point = torch.from_numpy(np.ascontiguousarray(uv)).float().clone()

        # --- Data completion and shape fixing ---

        # 1. Fill in mask and verts
        mask = torch.ones((self.size, self.size), dtype=torch.float32)
        verts = torch.zeros((778, 3), dtype=torch.float32)

        aug_param = torch.zeros((4), dtype=torch.float32)
        bb2img_trans = torch.zeros((2, 3), dtype=torch.float32)

        # 2. Fix root dimension (ensure it is [3])
        if xyz_root.dim() == 2 and xyz_root.shape[0] == 1:
            xyz_root = xyz_root.squeeze(0)

        # --- Core modification: adapt to FreiHAND contrastive mode ---
        if self.cfg.DATA.CONTRASTIVE and 'train' in self.phase:
            # FreiHAND concatenates two views along the channel dimension
            # Image: [3, H, W] cat [3, H, W] -> [6, 128, 128]
            img = torch.cat([img, img], 0)

            # Mask: [H, W] -> unsqueeze -> [1, H, W] -> cat -> [2, 128, 128]
            mask = mask.unsqueeze(0)
            mask = torch.cat([mask, mask], 0)

            # Joints/Verts: concatenate along the last dimension
            # Joint Img: [21, 2] -> [21, 4]
            joint_img = torch.cat([uv_point, uv_point], -1)
            # Joint Cam: [21, 3] -> [21, 6]
            joint_cam = torch.cat([xyz, xyz], -1)
            # Verts: [778, 3] -> [778, 6]
            verts = torch.cat([verts, verts], -1)

            # Calib: match FreiHAND contrastive shape by concatenating along dim=0 -> [8, 4]
            calib = torch.cat([calib, calib], 0)

            # Others
            aug_param = torch.cat([aug_param, aug_param], 0)      # [8]
            bb2img_trans = torch.cat([bb2img_trans, bb2img_trans], -1)  # [2, 6]

            root = torch.cat([xyz_root, xyz_root], -1)
            res = {
                'img': img,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'verts': verts,
                'mask': mask,
                'root': root,
                'calib': calib,
                'aug_param': aug_param,
                'bb2img_trans': bb2img_trans
            }
        else:
            # Normal mode
            res = {
                'img': img,
                'joint_img': uv_point,
                'joint_cam': xyz,
                'verts': verts,
                'mask': mask,
                'root': xyz_root,
                'calib': calib
            }

        return res

    def visualization(self, idx, data):
        gs = gridspec.GridSpec(1, 2)
        fig = plt.figure()
        ax = fig.add_subplot(gs[0, 0])

        # Visualization compatibility
        img_vis = data['img']
        if img_vis.shape[0] == 6:  # contrastive mode [6, 128, 128]
            img_vis = img_vis[:3, :, :]  # take the first 3 channels

        img = inv_base_tranmsform(img_vis.numpy(), std=self.img_std, mean=self.img_mean)
        ax.imshow(img)

        uv_vis = data['joint_img']
        if uv_vis.shape[-1] == 4:  # [21, 4]
            uv_vis = uv_vis[:, :2]

        uv_point = uv_vis.numpy() * img.shape[:2][::-1]
        ax.scatter(uv_point[:, 0], uv_point[:, 1])
        ax.axis('off')
        plt.show()

    def __len__(self):
        return len(self.image_paths)

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = list(results_pose_cam_xyz.keys())
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item()


if __name__ == '__main__':
    from main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    args.config_file = '/home/hdh/desk/zjj_Datas/Lite-AMNet/configs/Lite-AMNet.yml'
    cfg = setup(args)
    dataset = Ge(cfg, 'eval')
    for i in range(0, 10):
        data = dataset.__getitem__(i)
        dataset.visualization(i, data)

