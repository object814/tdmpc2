"""torch Dataset over the HDF5 produced by `collect_data.py`.

Returns per-sample dict observations in the exact format the TD-MPC2 encoder
consumes:
    state : (7,)      float32   raw proprio
    rgb   : (3N,H,W)  float32   raw pixels in [0,255] (the encoder's internal
                                PixelPreprocess does /255 - 0.5 itself)

The HDF5 file handle is opened lazily per worker process so the dataset is
safe under a multi-worker DataLoader.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainObsDataset(Dataset):
    def __init__(self, h5_path):
        import h5py
        self.h5_path = str(h5_path)
        self._h5 = None
        # Read length + metadata once up front, then close.
        with h5py.File(self.h5_path, "r") as f:
            self._len = int(f["state"].shape[0])
            self.image_size = int(f.attrs["image_size"])
            self.img_channels = int(f.attrs["img_channels"])
            self.num_cameras = int(f.attrs["num_cameras"])
            self.cameras = json.loads(f.attrs["cameras"])
            self.tasks = json.loads(f.attrs["tasks"])

    def _ensure_open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        f = self._ensure_open()
        state = torch.from_numpy(f["state"][idx].astype(np.float32))
        rgb = torch.from_numpy(f["rgb"][idx].astype(np.float32))  # [0,255]
        return {"state": state, "rgb": rgb}

    def __getstate__(self):
        # Don't pickle the open h5 handle across worker fork.
        d = self.__dict__.copy()
        d["_h5"] = None
        return d
