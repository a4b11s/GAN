from keras.api.layers import Conv2D, Layer, SpectralNormalization
from keras.api import ops

from typing import Any
from numpy import float64
from numpy.typing import NDArray
from optree import PyTree


class ResBlockDown(Layer):
    def __init__(self, filters, downsample=True):
        super(ResBlockDown, self).__init__()
        self.filters = filters
        self.downsample = downsample

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = SpectralNormalization(Conv2D(self.filters, 3, padding="same"))

        self.conv_2 = SpectralNormalization(Conv2D(self.filters, 3, padding="same"))

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = SpectralNormalization(
                Conv2D(
                    self.filters,
                    1,
                    padding="same",
                )
            )

    def down(self, x: PyTree | NDArray[Any]) -> PyTree | NDArray[float64]:
        return ops.average_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

    def call(self, input_tensor: Any, **kwargs) -> Any:
        x = self.conv_1(input_tensor)
        x = ops.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = ops.leaky_relu(x, 0.2)

        if self.downsample:
            x = self.down(x)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = ops.nn.leaky_relu(skip, 0.2)

            if self.downsample:
                skip = self.down(skip)
        else:
            skip = input_tensor

        output = skip + x

        return output
