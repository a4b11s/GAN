import math

from PIL import Image, ImageOps
from numpy.typing import NDArray


class ImagePlotter:
    def __init__(self, path: str | None = None) -> None:
        self.path = path

    def __call__(
        self, images: NDArray, gap: int = 1, save_path: str | None = None
    ) -> Image.Image:
        image_count = images.shape[0]

        image_size = images.shape[1]

        gaped_image_size = image_size + gap * 2

        cols_count = math.ceil(math.sqrt(image_count))
        rows_count = math.ceil(image_count / cols_count)

        image_plot = Image.new(
            "RGB",
            (
                gaped_image_size * cols_count - gap * 2,
                gaped_image_size * rows_count - gap * 2,
            ),
            (255, 255, 255),
        )

        for index, image in enumerate(images):
            row = index // cols_count
            col = index % cols_count

            x_offset = col * gaped_image_size
            y_offset = row * gaped_image_size

            image_from_array = Image.fromarray(image)
            image_plot.paste(image_from_array, (x_offset, y_offset))

        image_plot = ImageOps.expand(image_plot, border=gap * 2, fill=(255, 255, 255))

        if save_path is not None:
            self.save(image_plot, save_path)

        return image_plot

    def save(self, image: Image.Image, path: str | None = None) -> None:
        if path is None:
            path = self.path
        image.save(path)
