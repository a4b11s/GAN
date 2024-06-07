import numpy as np
from keras.api.activations import tanh
from keras.api.layers import (
    Input,
    Reshape,
    UpSampling2D,
    Conv2D,
    SpectralNormalization,
    BatchNormalization,
    LeakyReLU,
)
from keras.api.models import Model
from enum import Enum

from ganai.layers import ResBlock, SelfAttention

G_NORM = Enum("NORM", ["BN", "SN", "none"])


def build_generator(
    img_size: int,
    latent_dim: int,
    filters_start: int,
    layer_count: int,
    norm: G_NORM,
    att_layers_num: list[int],
) -> Model:
    """
    Builds a generator model for a GAN.

    Args:
        img_size (int): The size of the generated images.
        latent_dim (int): The dimensionality of the latent space.
        filters_start (int): The starting number of filters in the generator.
        layer_count (int): The number of layers in the generator.
        norm (G_NORM): The normalization method to use in the generator.
        att_layers_num (list[int]): A list of indices where self-attention layers should be added.

    Returns:
        Model: A Keras model representing the generator.
    """

    z = Input(shape=(latent_dim,))

    #TODO: add calculation of shape based on latent_dim and img_size
    x = Reshape((2, 2, int(latent_dim / (2**2))))(z)

    f_multiplayer = 2 ** (layer_count - 1)

    for i in range(layer_count):
        filters = filters_start * int(f_multiplayer)
        print(f"filters: {filters}, i: {i}, f_multiplayer: {f_multiplayer}")

        
        if norm == G_NORM.SN:
            x = SpectralNormalization(
                Conv2D(filters, 3, padding="same")
            )(x)
        else:
            x = Conv2D(filters, 3, padding="same")(x)
            
        x = UpSampling2D((2, 2))(x)

        if norm == G_NORM.BN:
            x = BatchNormalization()(x)

        x = LeakyReLU()(x)

        if i in att_layers_num and filters_start > 16:
            x = SelfAttention()(x)
            x = LeakyReLU()(x)

        f_multiplayer /= 2

    output_img = tanh(SpectralNormalization(Conv2D(3, 3, padding="same"))(x))

    return Model([z], output_img, name="generator")


# def build_sa_generator(
#     latent_dim: int,
#     filters_start: int,
#     filters_multiplayer: list[int],
#     is_attentions: list[bool],
# ) -> Model:
#     filters_size = np.array(filters_multiplayer) * filters_start


#     for c_filter, is_attention in zip(filters_size, is_attentions):
#         x = ResBlock(c_filter)(x)
#         x =

#         if is_attention:
#             x = SelfAttention()(x)
