from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FRAME_INDICES = np.linspace(0, 39, 30, dtype=int)
RADAR_LOG_MEAN = 3.03424282774987
RADAR_LOG_STD = 3.048960853873337
IMU_MEAN = np.array(
    [-0.0862426042, 0.8751159239, 0.3859220429, 0.0299630752, 0.0002065225, 0.0313615273],
    dtype=np.float32,
)
IMU_STD = np.array(
    [0.1163086134, 0.0845464994, 0.1936089863, 2.4873266666, 2.6552567616, 1.7453757561],
    dtype=np.float32,
)


class EpisodeDataset(Dataset):
    """Radar and IMU are one mandatory deployed sensor input."""

    def __init__(
        self,
        data_root,
        domain="source",
        protocol="official_non_ood",
        include_keypoints=True,
        include_labels=False,
    ):
        if domain not in {"source", "target"}:
            raise ValueError(f"domain must be source or target, got {domain}")
        if protocol not in {"official_non_ood", "person_1_8_to_9_12"}:
            raise ValueError(f"unknown protocol: {protocol}")
        self.data_root = Path(data_root)
        self.include_keypoints = include_keypoints
        self.include_labels = include_labels
        rows = []
        for split in ("train", "eval"):
            path = self.data_root / f"{split}_list.txt"
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                name, person, scene, label, trial = line.split(",")[:5]
                rows.append(
                    {
                        "filename": name,
                        "split": split,
                        "person": int(person),
                        "scene": int(scene),
                        "label": int(label) - 1,
                        "trial": int(trial),
                    }
                )
        if protocol == "official_non_ood":
            selected = "train" if domain == "source" else "eval"
            self.rows = [row for row in rows if row["split"] == selected]
        else:
            self.rows = [
                row
                for row in rows
                if (row["person"] <= 8) == (domain == "source")
            ]

    def __len__(self):
        return len(self.rows)

    def _load_keypoints(self, row):
        path = self.data_root / "keypoint_gt" / f"{row['filename']}.npy"
        keypoints = np.load(path).astype(np.float32)
        if keypoints.shape[1] == 21:
            hand = keypoints.reshape(30, 1, 21, 3)
            padding = np.zeros_like(hand)
            if row["person"] in {2, 4, 5, 12}:
                keypoints = np.concatenate((hand, padding), axis=1)
            else:
                keypoints = np.concatenate((padding, hand), axis=1)
        return keypoints.reshape(30, 2, 21, 3)

    def __getitem__(self, index):
        row = self.rows[index]
        name = row["filename"]
        with h5py.File(self.data_root / "mmwave" / f"{name}.mat", "r") as handle:
            radar = np.array(handle["data"], dtype=np.float32).transpose((2, 1, 0))
        radar = (np.log1p(radar[FRAME_INDICES]) - RADAR_LOG_MEAN) / RADAR_LOG_STD
        imu = np.load(self.data_root / "imu" / f"{name}.npy").astype(np.float32)
        imu = (imu[FRAME_INDICES] - IMU_MEAN) / IMU_STD
        item = {
            "radar": torch.from_numpy(radar).unsqueeze(1),
            "imu": torch.from_numpy(imu),
            "filename": name,
        }
        if self.include_keypoints:
            item["keypoints"] = torch.from_numpy(self._load_keypoints(row))
        if self.include_labels:
            item["label"] = torch.tensor(row["label"], dtype=torch.long)
        return item
