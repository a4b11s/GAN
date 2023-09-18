import numpy as np
from keras.layers import Flatten, Input, Dense
from keras.models import Model

from utilites.layers import ResBlockDown, SelfAttention


def build_discriminator(img_size, filters_start=16, filters_multiplayer=None, is_attentions=None):
    if filters_multiplayer is None:
        filters_multiplayer = [1, 2, 4]

    if is_attentions is None:
        is_attentions = [False, True, False]

    filters_size = np.array(filters_multiplayer) * filters_start

    img = Input(shape=(img_size, img_size, 3))

    x = img

    for c_filter, is_attention in zip(filters_size, is_attentions):
        x = ResBlockDown(c_filter)(x)  # 64
        if is_attention:
            x = SelfAttention()(x)

    x = ResBlockDown(filters_size[-1], False)(x)
    x = Flatten()(x)

    output = Dense(1)(x)

    return Model(img, output, name="discriminator")
