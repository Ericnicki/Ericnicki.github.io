from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FRAME_INDICES = np.linspace(0, 39, 30, dtype=int)
RADAR_LOG_MEAN = 3.03424282774987
RADAR_LOG_STD = 3.048960853873337


class RadarClassificationDataset(Dataset):
    """Official-split radar episodes for direct gesture classification."""

    def __init__(self, data_root, split):
        if split not in {"train", "eval"}:
            raise ValueError(f"split must be train or eval, got {split}")
        self.data_root = Path(data_root)
        self.rows = []
        for line in (self.data_root / f"{split}_list.txt").read_text().splitlines():
            if not line.strip():
                continue
            name, person, scene, label, trial = line.split(",")[:5]
            self.rows.append(
                {
                    "filename": name,
                    "person": int(person),
                    "scene": int(scene),
                    "label": int(label) - 1,
                    "trial": int(trial),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with h5py.File(
            self.data_root / "mmwave" / f"{row['filename']}.mat", "r"
        ) as handle:
            radar = np.array(handle["data"], dtype=np.float32).transpose((2, 1, 0))
        radar = (np.log1p(radar[FRAME_INDICES]) - RADAR_LOG_MEAN) / RADAR_LOG_STD
        return {
            "radar": torch.from_numpy(radar).unsqueeze(1),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "filename": row["filename"],
        }
