import numpy as np
from keras.api.callbacks import Callback

from ganai.utilites import ImagePlotter


class GANMonitor(Callback):
    def __init__(
        self,
        batch_size: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.batch_size: int = batch_size
        self.seed: int | None = seed
        self.plotter = ImagePlotter()

    def on_epoch_end(self, epoch: int, logs: tuple[str] | None = None) -> None:
        image_batchs, seed = self.model.generate(self.batch_size, seed=self.seed)

        for index, batch in enumerate(image_batchs):
            plotted_image = self.plotter((batch * 255).astype(np.uint8))

            plotted_image.save(f"./data/output/e-{epoch + 1}-bc-{index + 1}-seed-{seed}.png")
