import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from torchgeo.datasets.geo import NonGeoDataset
from transformers import AutoProcessor

from .transforms import get_pretrained_gsv_train_transform


CHECK_MIN_FILESIZE = 5000  # 5 KB for GSV images

class GSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/gsv",
        batch_size: int = 64,
        num_workers: int = 6,
        crop_size: int = 224,
        val_random_split_fraction: float = 0.1,
        transform: str = 'gsv_transform',
        mode: str = "both",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform == 'gsv_transform':
            image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.train_transform = get_pretrained_gsv_train_transform(image_processor)
        else:
            self.train_transform = transform
        self.val_random_split_fraction = val_random_split_fraction
        self.mode = mode
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            print(f"Dataset path {self.data_dir} not found.")

    def setup(self, stage="fit"):
        dataset = GSVDataset(root=self.data_dir, transform=self.train_transform, mode=self.mode)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = random_split(dataset, [N_train, N_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError

class GSVDataset(NonGeoDataset):
    validation_filenames = [
        "index.csv",
        "images/",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        limit: Optional[int] = 10000,
    ) -> None:
        assert mode in ["both", "points"]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.limit = limit
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        index_fn = os.path.join(self.root, "index.csv")
        df = pd.read_csv(index_fn)

        self.filenames = []
        self.points = []

        n_skipped_files = 0
        n_loaded_files = 0
        for i in range(df.shape[0]):
            if self.limit is not None and n_loaded_files >= self.limit:
                break

            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])
            if not os.path.exists(filename) or os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(filename)
            self.points.append((df.iloc[i]["longitude"], df.iloc[i]["latitude"]))
            n_loaded_files += 1

        print(f"Skipped {n_skipped_files}/{len(df)} images due to small file size.")
        print(f"Loaded {len(self.filenames)}/{min(len(df), self.limit)} images.")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        point = torch.tensor(self.points[index])
        sample = {"point": point}

        if self.mode == "both":
            image = Image.open(self.filenames[index])
            # image = Image.open(self.filenames[index]).convert("RGB")
            # image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
            # image = torch.tensor(image).permute(2, 0, 1)  # Change to [C, H, W]
            sample["image"] = image

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(f"Missing file: {filepath}")
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        image = sample["image"].numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
