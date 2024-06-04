import math
import os
import random
from tkinter import Image
import numpy as np

from keras.api.utils import img_to_array, load_img, Sequence
from typing import Literal, Any

MODE = Literal["one", "two"]


class LoadItem(Sequence):
    def __init__(
        self,
        dataset_directory: str,
        img_size: int,
        batch_size: int,
        mode: MODE = "two",
    ):
        self.dataset_directory: str = dataset_directory
        self.batch_size: int = batch_size
        self.img_size: int = img_size
        self.file_list: list[str] = self._get_data_list()
        self.mode: MODE = mode
        print(f"detected {len(self.file_list)} pictures")

    def _get_data_list(self) -> list[str]:
        subdirs: list[str] = os.listdir(self.dataset_directory)
        files: list[str] = []

        for subdir in subdirs:
            files_in_subdir = os.listdir(f"{self.dataset_directory}/{subdir}")
            files_with_subdir = [
                f"{self.dataset_directory}/{subdir}/{f}" for f in files_in_subdir
            ]
            files += files_with_subdir

        return files

    def __len__(self) -> int:
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[list[Any], list[Any]]
        | tuple[list[Any]]
    ):
        low = idx * self.batch_size
        high = low + self.batch_size
        batch_paths = self.file_list[low:high]
        batch = []
        for path in batch_paths:
            img: Image = load_img(
                path,
                color_mode="rgb",
                target_size=(self.img_size, self.img_size),
                interpolation="bicubic",
            )
            img_array = img_to_array(img)
            img_array = img_array / 255
            batch.append(img_array)
        batch = np.array(batch)
        if batch.shape != (self.batch_size, self.img_size, self.img_size, 3):
            print("\n")
            print(low)
            print(high)
            print(batch.shape)

        if self.mode == "two":
            y = batch, batch
        else:
            y = batch
        return y

    def on_epoch_end(self) -> None:
        random.shuffle(self.file_list)
