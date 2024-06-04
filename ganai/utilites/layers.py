import tensorflow as tf
from keras.api.constraints import Constraint
from keras.api.layers import Conv2D, Layer


class SpectralNormalization(Constraint):
    def __init__(self, n_iter=5):
        # n_iter is a hyperparam
        self.n_iter = n_iter

    def call(self, input_weights):
        # reshape weights from conv layer into (HxW, C)
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))

        u = tf.random.normal((w.shape[0], 1))

        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)

            u = tf.matmul(w, v)
            u /= tf.norm(u)

        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)

        return input_weights / spec_norm


class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        n, h, w, c = input_shape
        self.n_feats = h * w
        self.conv_theta = Conv2D(c // 8, 1,
                                 padding='same',
                                 kernel_constraint=SpectralNormalization(),
                                 name='Conv_Theta')

        self.conv_phi = Conv2D(c // 8, 1,
                               padding='same',
                               kernel_constraint=SpectralNormalization(),
                               name='Conv_Phi')

        self.conv_g = Conv2D(c // 2, 1,
                             padding='same',
                             kernel_constraint=SpectralNormalization(),
                             name='Conv_G')

        self.conv_attn_g = Conv2D(c, 1,
                                  padding='same',
                                  kernel_constraint=SpectralNormalization(),
                                  name='Conv_AttnG')

        self.sigma = self.add_weight(shape=[1],
                                     initializer='zeros',
                                     trainable=True,
                                     name='sigma')

    def call(self, x, **kwargs):
        # theta => key
        # phi => query
        # g => query

        n, h, w, c = x.shape

        theta = self.conv_theta(x)
        theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))

        phi = self.conv_phi(x)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
        phi = tf.reshape(phi, (-1, self.n_feats // 4, phi.shape[-1]))

        # generate attention map
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
        g = tf.reshape(g, (-1, self.n_feats // 4, g.shape[-1]))

        # multiply attn map with feature maps
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
        attn_g = self.conv_attn_g(attn_g)

        output = x + self.sigma * attn_g

        return output


class ResBlockDown(Layer):
    def __init__(self, filters, downsample=True):
        super(ResBlockDown, self).__init__()
        self.filters = filters
        self.downsample = downsample

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding="same", kernel_constraint=SpectralNormalization())

        self.conv_2 = Conv2D(self.filters, 3, padding="same", kernel_constraint=SpectralNormalization())

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding="same", kernel_constraint=SpectralNormalization())

    def down(self, x):
        return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

    def call(self, input_tensor, **kwargs):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.downsample:
            x = self.down(x)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = tf.nn.leaky_relu(skip, 0.2)

            if self.downsample:
                skip = self.down(skip)
        else:
            skip = input_tensor

        output = skip + x

        return output


class ResBlock(Layer):
    def __init__(self, filters):
        super(ResBlock, self).__init__(name=f"g_resblock_{filters}x{filters}")
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding="same", name="conv2d_1",
                             kernel_constraint=SpectralNormalization())

        self.conv_2 = Conv2D(self.filters, 3, padding="same", name="conv2d_2",
                             kernel_constraint=SpectralNormalization())

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding="same", name="conv2d_3",
                                 kernel_constraint=SpectralNormalization())

    def call(self, input_tensor, **kwargs):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = tf.nn.leaky_relu(skip, 0.2)
        else:
            skip = input_tensor

        output = skip + x
        return output
