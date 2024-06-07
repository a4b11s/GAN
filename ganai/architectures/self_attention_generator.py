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
    filters_start: int,
    filters_multiplayer: list[int],
    is_attentions: list[bool],
) -> Model:
    filters_size = np.array(filters_multiplayer) * filters_start

    z = Input(shape=(latent_dim,))

    x = Reshape((2, 2, int(latent_dim / (2**2))))(z)

    for c_filter, is_attention in zip(filters_size, is_attentions):
        x = ResBlock(c_filter)(x)
        x = UpSampling2D((2, 2))(x)

        if is_attention:
            x = SelfAttention()(x)

    output_img = tanh(SpectralNormalization(Conv2D(3, 3, padding="same"))(x))

    return Model([z], output_img, name="generator")
