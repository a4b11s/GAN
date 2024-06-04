from keras.api.layers import Conv2D, Layer, SpectralNormalization
from keras.api import ops


class ResBlock(Layer):
    def __init__(self, filters):
        super(ResBlock, self).__init__(name=f"g_resblock_{filters}x{filters}")
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = SpectralNormalization(
            Conv2D(
                self.filters,
                3,
                padding="same",
                name="conv2d_1",
            )
        )

        self.conv_2 = SpectralNormalization(
            Conv2D(
                self.filters,
                3,
                padding="same",
                name="conv2d_2",
            )
        )

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = SpectralNormalization(
                Conv2D(
                    self.filters,
                    1,
                    padding="same",
                    name="conv2d_3",
                )
            )

    def call(self, input_tensor, **kwargs):
        x = self.conv_1(input_tensor)
        x = ops.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = ops.leaky_relu(x, 0.2)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = ops.leaky_relu(skip, 0.2)
        else:
            skip = input_tensor

        output = skip + x
        return output
