from keras.api.layers import Conv2D, Layer, SpectralNormalization, MaxPool2D
from keras.api import ops

import tensorflow as tf


class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.n_feats = h * w
        self.conv_theta = Conv2D(
            c // 8,
            1,
            padding="same",
            kernel_constraint=SpectralNormalization(),
            name="Conv_Theta",
        )

        self.conv_phi = Conv2D(
            c // 8,
            1,
            padding="same",
            kernel_constraint=SpectralNormalization(),
            name="Conv_Phi",
        )

        self.conv_g = Conv2D(
            c // 2,
            1,
            padding="same",
            kernel_constraint=SpectralNormalization(),
            name="Conv_G",
        )
        
        self.conv_attn_g = Conv2D(
            c,
            1,
            padding="same",
            kernel_constraint=SpectralNormalization(),
            name="Conv_AttnG",
        )
        
        self.max_pull_2d = MaxPool2D(pool_size=2, strides=2, padding="VALID", name="Max_Pull_2D")

        self.sigma = self.add_weight(
            shape=[1], initializer="zeros", trainable=True, name="sigma"
        )

    def call(self, x, **kwargs):
        # theta => key
        # phi => query
        # g => query

        n, h, w, c = x.shape

        theta = self.conv_theta(x)
        theta = ops.reshape(theta, (-1, self.n_feats, theta.shape[-1]))

        phi = self.conv_phi(x)
        phi = ops.max_pool(phi, pool_size=2, strides=2, padding="VALID")
        phi = ops.reshape(phi, (-1, self.n_feats // 4, phi.shape[-1]))

        # generate attention map
        attn = ops.matmul(theta, ops.transpose(phi))
        attn = ops.softmax(attn)

        g = self.conv_g(x)
        g = self.max_pull_2d(g)
        g = ops.reshape(g, (-1, self.n_feats // 4, g.shape[-1]))

        # multiply attn map with feature maps
        attn_g = ops.matmul(attn, g)
        attn_g = ops.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
        attn_g = self.conv_attn_g(attn_g)

        output = x + self.sigma * attn_g

        return output
