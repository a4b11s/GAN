import tensorflow as tf
from keras.api.utils import image_dataset_from_directory


class DatasetFromDir:
    def __init__(
        self, dir: str, image_size: int, batch_size: int, split: float
    ) -> None:
        self.dir = dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.split = split

    def load_dataset(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_dataset: tf.data.Dataset = image_dataset_from_directory(
            directory=self.dir,
            seed=271202,
            batch_size=self.batch_size,
            label_mode=None,
            color_mode="rgb",
            image_size=(self.image_size, self.image_size),
            interpolation="bicubic",
            validation_split=self.split,
            subset="training",
        )
        val_data: tf.data.Dataset = image_dataset_from_directory(
            directory=self.dir,
            seed=271202,
            batch_size=self.batch_size,
            label_mode=None,
            color_mode="rgb",
            image_size=(self.image_size, self.image_size),
            interpolation="bicubic",
            validation_split=self.split,
            subset="validation",
        )
        
        buffer_size = self.batch_size * 4

        train_dataset = train_dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=buffer_size)
        val_data = val_data.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=buffer_size)

        return train_dataset, val_data

    def preprocess_image(self, data):
        data = tf.cast(data / 255.0, tf.float32)
        return data
