import numpy as np
from keras.api.activations import tanh
from keras.api.layers import (
    Input,
    Reshape,
    UpSampling2D,
    Conv2D,
    SpectralNormalization,
)
from keras.api.models import Model

from ganai.layers import ResBlock, SelfAttention


def build_sa_generator(
    latent_dim: int,
    filters_start: int = 16,
    filters_multiplayer: list[int] = None,
    is_attentions: list[bool] = None,
) -> Model:
    if filters_multiplayer is None:
        filters_multiplayer = [8, 4, 2, 1]

    if is_attentions is None:
        is_attentions = [False, False, False, False]

    filters_size = np.array(filters_multiplayer) * filters_start

    z = Input(shape=(latent_dim,))

    x = Reshape((4, 4, int(latent_dim / (4 * 4))))(z)

    for c_filter, is_attention in zip(filters_size, is_attentions):
        x = UpSampling2D((2, 2))(x)
        x = ResBlock(c_filter)(x)

        if is_attention:
            x = SelfAttention()(x)

    output_img = tanh(SpectralNormalization(Conv2D(3, 3, padding="same"))(x))

    return Model([z], output_img, name="generator")
