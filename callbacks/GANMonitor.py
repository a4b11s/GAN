import numpy as np
from keras.callbacks import Callback

from utilites.plot_image import plot_image


class GANMonitor(Callback):
    def __init__(self, batch_size, is_random_seed=None, fixed_seed=None):
        super().__init__()
        self.batch_size = batch_size
        self.is_random_seed = is_random_seed
        self.fixed_seed = fixed_seed

        if fixed_seed is None and is_random_seed is None:
            self.is_random_seed = True

    def on_epoch_end(self, epoch, logs=None):
        if self.is_random_seed:
            image_batchs, seed = self.model.generate(self.batch_size)

        if self.fixed_seed is not None:
            image_batchs, seed = self.model.generate(self.batch_size, seed=self.fixed_seed)

        for index, batch in enumerate(image_batchs):
            plotted_image = plot_image((batch * 255).astype(np.uint8))

            plotted_image.save(f"output/e-{epoch + 1}-bc-{index + 1}-seed-{seed}.png")
