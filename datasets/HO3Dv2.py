import sys
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
import pickle
from utils.preprocessing import augmentation, augmentation_2d
from utils.augmentation import Augmentation
from utils.vis import base_transform
from build import DATA_REGISTRY
from termcolor import cprint


@DATA_REGISTRY.register()
class HO3Dv2(data.Dataset):

    def __init__(self, cfg, phase='train', writer=None):
        super(HO3Dv2, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.data_path = self.cfg.DATA.HO3Dv2.ROOT

        # Load index file
        self.item_list = self._load_file_list(self.data_path, self.phase)

        self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None
        self.one_version_len = len(self.item_list)

        # Debug flag
        self.has_printed_debug = False

        if writer is not None:
            writer.print_str(f'Loaded HO3Dv2 {self.phase} {len(self.item_list)} samples')
        cprint(f'Loaded HO3Dv2 {self.phase} {len(self.item_list)} samples', 'red')

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        # Process index
        real_idx = idx % len(self.item_list)

        try:
            if 'train' in self.phase:
                return self.get_training_sample(real_idx)
            elif 'eval' in self.phase:
                return self.get_eval_sample(real_idx)
            else:
                raise Exception('Phase error')
        except Exception as e:
            # Fault tolerance: if an image is corrupted, print error and randomly return another one
            print(f"[HO3Dv2 Error] Failed to load index {real_idx}: {e}")
            # Randomly return a substitute sample to prevent training interruption
            return self.__getitem__(np.random.randint(0, len(self.item_list)))

    def _load_file_list(self, data_path, phase):
        if 'train' in phase:
            txt_file = os.path.join(data_path, 'train.txt')
        else:
            txt_file = os.path.join(data_path, 'evaluation.txt')
            if not os.path.exists(txt_file):
                txt_file = os.path.join(data_path, 'eval.txt')

        print(f"Loading file list from: {txt_file}")

        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Missing index file: {txt_file}")

        items = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('/')
                if len(parts) >= 2:
                    items.append({'seq': parts[0], 'id': parts[1]})
        return items

    def _load_data_on_the_fly(self, idx):
        item = self.item_list[idx]
        seq = item['seq']
        fid = item['id']
        base_folder = 'train' if 'train' in self.phase else 'evaluation'

        # 1. Try to load image
        img_path_jpg = os.path.join(self.data_path, base_folder, seq, 'rgb', f"{fid}.jpg")
        img_path_png = os.path.join(self.data_path, base_folder, seq, 'rgb', f"{fid}.png")

        img_path = img_path_jpg
        if not os.path.exists(img_path) and os.path.exists(img_path_png):
            img_path = img_path_png

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # 2. Load meta (Pickle)
        meta_path = os.path.join(self.data_path, base_folder, seq, 'meta', f"{fid}.pkl")

        # Default data initialization
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 600.0
        K[1, 1] = 600.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        xyz = np.zeros((21, 3), dtype=np.float32)
        verts = np.zeros((778, 3), dtype=np.float32)
        bbox = [0, 0, img.shape[1], img.shape[0]]

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta_data = pickle.load(f, encoding='latin1')

                if 'camMat' in meta_data:
                    K = meta_data['camMat'].astype(np.float32)
                if 'handJoints3D' in meta_data:
                    xyz = meta_data['handJoints3D'].astype(np.float32)

                # Strictly process vertices
                if 'handMeshVerts3D' in meta_data:
                    raw_verts = meta_data['handMeshVerts3D']
                    if hasattr(raw_verts, 'shape') and raw_verts.shape == (778, 3):
                        verts = raw_verts.astype(np.float32)

                if 'handBoundingBox' in meta_data:
                    bbox = meta_data['handBoundingBox']

                # Recompute bounding box based on joints (more robust)
                pts_proj = np.dot(K, xyz.T).T
                depth = pts_proj[:, 2:]
                depth[depth == 0] = 1e-5
                pts_proj = pts_proj[:, :2] / depth
                min_coords = np.min(pts_proj, axis=0)
                max_coords = np.max(pts_proj, axis=0)
                bbox = [min_coords[0], min_coords[1],
                        max_coords[0] - min_coords[0],
                        max_coords[1] - min_coords[1]]

                # Enlarge bounding box
                scale = 1.2
                c_x, c_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                bbox[2] *= scale
                bbox[3] *= scale
                bbox[0] = c_x - bbox[2] / 2
                bbox[1] = c_y - bbox[3] / 2

            except Exception as e:
                print(f"Warning: pickle load error {meta_path}: {e}")

        # 3. Mask (HO3Dv2 does not provide masks, generate an all-white mask)
        # Ensure mask is single-channel (H, W)
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

        return {
            "img": img,
            "verts": verts,
            "xyz": xyz,
            "K": K,
            "bbox": bbox,
            "mask": mask
        }

    def get_training_sample(self, idx):
        raw_data = self._load_data_on_the_fly(idx)

        img = raw_data['img']
        vert = raw_data['verts']
        joint_cam = raw_data['xyz']
        K = raw_data['K']
        bbox = raw_data['bbox']
        mask = raw_data['mask']

        focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
        princpt = K[0:2, 2].astype(np.float32)

        # Data augmentation
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(
            img, bbox, self.phase,
            exclude_flip=not self.cfg.DATA.HO3Dv2.FLIP,
            input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
            mask=mask,
            base_scale=self.cfg.DATA.HO3Dv2.BASE_SCALE,
            scale_factor=self.cfg.DATA.HO3Dv2.SCALE,
            rot_factor=self.cfg.DATA.HO3Dv2.ROT,
            shift_wh=[bbox[2], bbox[3]],
            gaussian_std=self.cfg.DATA.STD)

        if self.color_aug is not None:
            roi = self.color_aug(roi)

        roi = base_transform(roi, self.cfg.DATA.SIZE,
                             mean=self.cfg.DATA.IMG_MEAN,
                             std=self.cfg.DATA.IMG_STD)

        # === Critical fix: use torch.tensor() instead of torch.from_numpy() ===
        # torch.tensor() creates a copy and completely avoids memory-sharing issues
        roi = torch.tensor(roi, dtype=torch.float32)

        # Mask processing: ensure it is 2D (128, 128)
        # If augmentation returns (128,128,1) or (128,128), unify to (128,128)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = torch.tensor(mask, dtype=torch.float32)

        # 2D joints
        joint_img_raw = np.dot(K, joint_cam.T).T
        joint_img_raw = joint_img_raw[:, :2] / (joint_img_raw[:, 2:] + 1e-5)

        aug_result = augmentation_2d(img, joint_img_raw, princpt, img2bb_trans, do_flip)
        if isinstance(aug_result, (tuple, list)):
            joint_img = aug_result[0]
        else:
            joint_img = aug_result

        joint_img = torch.tensor(joint_img[:, :2], dtype=torch.float32) / self.cfg.DATA.SIZE

        # 3D augmentation
        rot = aug_param[0]
        rot_aug_mat = np.array([
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)),  np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        joint_cam = np.dot(rot_aug_mat, joint_cam.T).T
        vert = np.dot(rot_aug_mat, vert.T).T

        # Camera calibration
        focal = focal * roi.size(1) / (bbox[2] * aug_param[1])
        calib = np.eye(4, dtype=np.float32)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.tensor(calib, dtype=torch.float32)

        # Post-processing
        root = joint_cam[0].copy()
        joint_cam -= root
        vert -= root
        joint_cam /= 0.2
        vert /= 0.2

        root = torch.tensor(root, dtype=torch.float32)
        joint_cam = torch.tensor(joint_cam, dtype=torch.float32)
        vert = torch.tensor(vert, dtype=torch.float32)

        # DEBUG print (only once)
        if not self.has_printed_debug:
            print("\n[HO3D DEBUG] Sample Data Shapes (Training):")
            print(f"  img: {roi.shape}")
            print(f"  mask: {mask.shape}")
            print(f"  joint_img: {joint_img.shape}")
            print(f"  verts: {vert.shape}")
            print(f"  calib: {calib.shape}")
            self.has_printed_debug = True

        return {
            'img': roi,
            'joint_img': joint_img,
            'joint_cam': joint_cam,
            'verts': vert,
            'mask': mask,
            'root': root,
            'calib': calib
        }

    def get_eval_sample(self, idx):
        raw_data = self._load_data_on_the_fly(idx)
        img = raw_data['img']
        K = raw_data['K']
        bbox = raw_data['bbox']

        focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
        princpt = K[0:2, 2].astype(np.float32)

        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, _ = augmentation(
            img, bbox, self.phase,
            exclude_flip=not self.cfg.DATA.HO3Dv2.FLIP,
            input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
            mask=None,
            base_scale=self.cfg.DATA.HO3Dv2.BASE_SCALE,
            scale_factor=self.cfg.DATA.HO3Dv2.SCALE,
            rot_factor=self.cfg.DATA.HO3Dv2.ROT,
            shift_wh=[bbox[2], bbox[3]],
            gaussian_std=self.cfg.DATA.STD)

        roi = base_transform(roi, self.cfg.DATA.SIZE,
                             mean=self.cfg.DATA.IMG_MEAN,
                             std=self.cfg.DATA.IMG_STD)
        roi = torch.tensor(roi, dtype=torch.float32)

        calib = np.eye(4, dtype=np.float32)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.tensor(calib, dtype=torch.float32)

        return {'img': roi, 'calib': calib}

