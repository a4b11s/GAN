from keras.api.layers import (
    Flatten,
    Input,
    Dense,
    Conv2D,
    SpectralNormalization,
    BatchNormalization,
    LeakyReLU,
)
from keras.api.models import Model
from enum import Enum

from ganai.layers import ResBlockDown, SelfAttention

D_NORM = Enum("NORM", ["BN", "SN", "none"])


def build_discriminator(
    img_size: int,
    filters_start: int,
    layer_count: int,
    norm: D_NORM,
    att_layers_num: list[int],
) -> Model:
    """
    Builds a discriminator model for a Generative Adversarial Network (GAN).

    Parameters:
    - img_size (int): The size of the input images (assumed to be square).
    - filters_start (int): The number of filters to start with in the first convolutional layer.
    - layer_count (int): The number of convolutional layers in the discriminator.
    - norm (D_NORM): The type of normalization to use (BN for Batch Normalization, SN for Spectral Normalization, or None).
    - att_layers_num (list[int]): A list of layer indices where Self-Attention layers should be added.

    Returns:
    - Model: The discriminator model.
    """
    img = Input(shape=(img_size, img_size, 3))

    x = img

    f_multiplayer = 1

    for i in range(layer_count):
        filters = filters_start * f_multiplayer

        if norm == D_NORM.SN:
            x = SpectralNormalization(
                Conv2D(filters, 3, strides=(2, 2), padding="same")
            )(x)
        else:
            x = Conv2D(filters, 3, strides=(2, 2), padding="same")(x)

        if norm == D_NORM.BN:
            x = BatchNormalization()(x)

        x = LeakyReLU()(x)

        if i in att_layers_num and filters_start > 16: 
            x = SelfAttention()(x)
            x = LeakyReLU()(x)

        f_multiplayer *= 2

    x = Flatten()(x)

    output = Dense(1)(x)

    return Model(img, output, name="discriminator")
