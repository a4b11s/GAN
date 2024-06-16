import os
import tensorflow as tf
import keras
from keras.api.utils import image_dataset_from_directory

from typing import Literal


class DatasetFromDir:
    def __init__(
        self,
        path: str,
        image_size: int,
        batch_size: int,
        split: float,
        seed: int = 271202,
        buffer_size: int = 5,
        cache_dir: str = ".cache/",
    ) -> None:
        """
        Initializes the DatasetFromDir class with the specified parameters.

        Parameters:
        path (str): The directory containing the images.
        image_size (int): The size of the images to be loaded.
        batch_size (int): The batch size for the dataset.
        split (float): The proportion of the dataset to be used for validation.
        seed (int, optional): A random seed for reproducibility. Defaults to 271202.
        buffer_size (int, optional): The buffer size for data prefetching. Defaults to 20.
        cache_dir (str, optional): The directory for caching the datasets. Defaults to "/.cache/".
        """

        self.path = path
        self.image_size = image_size
        self.batch_size = batch_size
        self.split = split

        self.seed = seed
        self.cache_dir = cache_dir
        self.buffer_size = buffer_size

        self.cache_full_path = os.path.join(self.path, self.cache_dir)
        self.subsets_cache_paths = {
            "training": os.path.join(self.cache_full_path, "train"),
            "validation": os.path.join(self.cache_full_path, "validation"),
        }

        self.d_policy = keras.mixed_precision.global_policy()
        self.is_float16 = self.d_policy.name == "mixed_float16"
        self.d_type = tf.float16 if self.is_float16 else tf.float32
    
    def __call__(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        This method is the entry point for the DatasetFromDir class. It checks if the cache exists,
        if not, it creates the cache by calling the _create_cache method. Then it loads the cache
        by calling the _load_cache method.

        Parameters:
        None

        Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the training and validation datasets.
        """

        train = self._load_dataset("training")
        validation = self._load_dataset("validation")

        return train, validation

    def _load_dataset(
        self, subset: Literal["training", "validation"]
    ) -> tf.data.Dataset:
        """
        This method loads the specified subset (either 'training' or 'validation') of images from the directory specified by the path attribute.

        Parameters:
        subset (Literal["training", "validation"]): The subset of images to load. Can be either 'training' or 'validation'.

        Returns:
        tf.data.Dataset: A TensorFlow Dataset object containing the loaded images.

        The dataset is first created using the `tf.keras.preprocessing.image_dataset_from_directory` function. It is then processed by applying a batch operation, a map operation to preprocess the images, and a prefetch operation to optimize data loading.

        The `image_dataset_from_directory` function is used with the following parameters:
            - directory: The directory containing the images.
            - seed: A random seed for reproducibility.
            - label_mode: Set to 'none' to disable label loading.
            - color_mode: Set to 'rgb' to load RGB images.
            - image_size: The size of the images to be loaded.
            - interpolation: Set to 'bicubic' for high-quality image resizing.
            - validation_split: The proportion of the dataset to be used for validation.
            - subset: The subset of images to be loaded.

        The dataset is then processed by applying the following operations:
            - batch: Batch the images into batches of the specified batch size.
            - map: Apply the preprocessing function to each image in the dataset.
            - prefetch: Optimize data loading by prefetching data into a buffer.

        The preprocessing function `_preprocess_image` is applied to each image in the dataset. It scales the image pixel values to the range [0, 1] and converts them to floating-point numbers.

        The resulting dataset is then returned.
        """

        dataset: tf.data.Dataset = image_dataset_from_directory(
            label_mode=None,
            color_mode="rgb",
            interpolation="bicubic",
            directory=self.path,
            seed=self.seed,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            validation_split=self.split,
            subset=subset,
        )

        dataset = dataset.map(
            self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(buffer_size=self.buffer_size)

        return dataset

    def _preprocess_image(self, data):
        """
        This method preprocesses each image in the dataset. It scales the image pixel values to the range [0, 1] and converts them to floating-point numbers.

        Parameters:
        data (tf.Tensor): A tensor containing the image data.

        Returns:
        tf.Tensor: A tensor containing the preprocessed image data.

        The pixel values of the image are divided by 255 and then cast to floating-point numbers using tf.cast.
        """

        data = tf.cast(data / 255.0, self.d_type)
        return data
