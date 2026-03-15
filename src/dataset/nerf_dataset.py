import numpy as np

import torch
from torch.utils.data import Dataset


class RayDataset(Dataset):
    def __init__(self, ray_path):
        data = np.load(ray_path)
        self.rays_o = data["rays_o"].astype(np.float32)
        self.rays_d = data["rays_d"].astype(np.float32)
        self.rgb = data["rgb"].astype(np.float32)
        self.masks = data["masks"].astype(np.float32)
        self.near = data["near"].astype(np.float32)
        self.far = data["far"].astype(np.float32)
        self.K = data["K"]
        self.H = int(data["H"])
        self.W = int(data["W"])

    def __len__(self):
        return len(self.rays_o)

    def __getitem__(self, idx):
        # return (
        #     self.rays_o[idx],
        #     self.rays_d[idx],
        #     self.masks[idx],
        #     self.near[idx],
        #     self.far[idx],
        # )
        return {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "rgb": self.rgb[idx],
            "mask": self.masks[idx],
            "near": self.near[idx],
            "far": self.far[idx],
        }
