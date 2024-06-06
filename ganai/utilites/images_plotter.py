import math

from PIL import Image, ImageOps
from numpy.typing import NDArray


class ImagePlotter:
    def __init__(self, path: str | None = None) -> None:
        """
        Initializes the ImagePlotter object.

        Args:
            path (str | None, optional): The default path to save the resulting image.
        """
        self.path = path

    def __call__(
        self, images: NDArray, gap: int = 1, save_path: str | None = None
    ) -> Image.Image:
        """
        Plots the given images in a grid layout with a specified gap between them.

        Args:
            images (NDArray): A numpy array with shape (image_count, image_size, image_size, 3) containing the images to be plotted.
            gap (int, optional): The gap between the images in the grid layout. Defaults to 1.
            save_path (str, optional): The path to save the resulting image. If None, the image will not be saved. Defaults to None.

        Returns:
            Image.Image: The resulting image with the plotted images in a grid layout.

        Example:
            >>> image_plotter = ImagePlotter()
            >>> images = np.array([image1, image2, image3])  # Replace with your images
            >>> plotted_image = image_plotter(images, gap=5)
            >>> image_plotter.save(plotted_image, "./data/img.png")
        """
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
        """
        Saves the given image to the specified path.

        Args:
            image (Image.Image): The image to be saved.
            path (str | None): The path to save the image. If None, the path from the ImagePlotter instance will be used.

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the path is None and the instance path is also None.

        Example:
            >>> image_plotter = ImagePlotter()
            >>> image = Image.open("example.png")
            >>> image_plotter.save(image, "./output.png")
        """
        if path is None:
            path = self.path
        image.save(path)
