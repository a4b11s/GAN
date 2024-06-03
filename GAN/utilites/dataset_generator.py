import math
import os
import random

import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img


class LoadItem(tf.keras.utils.Sequence):
    def __init__(self, dataset_directory, img_size, batch_size, mode='two'):
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.file_list = self.__get_data_list__()
        self.mode = mode
        print(f'detected {len(self.file_list)} pictures')

    def __get_data_list__(self):
        subdirs = os.listdir(self.dataset_directory)
        files = []

        for subdir in subdirs:
            files_in_subdir = os.listdir(f"{self.dataset_directory}/{subdir}")
            files_with_subdir = [f"{self.dataset_directory}/{subdir}/{f}" for f in files_in_subdir]
            files += files_with_subdir

        return files

    def __preprocess__(self, images):
        return images / 255

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = low + self.batch_size
        batch_paths = self.file_list[low:high]
        batch = []
        for path in batch_paths:
            img = load_img(path, color_mode="rgb", target_size=(self.img_size, self.img_size), interpolation="bicubic")
            img_array = img_to_array(img)
            img_array = img_array / 255
            batch.append(img_array)
        batch = np.array(batch)
        if batch.shape != (self.batch_size, self.img_size, self.img_size, 3):
            print('\n')
            print(low)
            print(high)
            print(batch.shape)

        if self.mode == 'two':
            y = batch, batch
        else:
            y = batch
        return y

    def on_epoch_end(self):
        random.shuffle(self.file_list)
