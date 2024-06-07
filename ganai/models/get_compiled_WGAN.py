import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.optimizers.schedules import CosineDecay
from keras.api.models import Model

from ganai.architectures import build_discriminator, build_generator, D_NORM, G_NORM

from ganai.losses import discriminator_loss, generator_loss
from ganai.models import WGAN
from ganai.utilites import KID


def get_compiled_wgan(
    img_size: int,
    latent_dim: int,
    kid_image_size: int,
    d_config: tuple[int, int, list[bool]],
    g_config: tuple[int, int, list[bool]],
) -> WGAN:
    """
    This function compiles a Wasserstein GAN (WGAN) model with specified discriminator and generator configurations.

    Args:
        img_size (int): The size of the input images.
        latent_dim (int): The dimensionality of the latent space.
        kid_image_size (int): The size of the images used for KID evaluation.
        d_config (tuple[int, int, list[bool]]): A tuple containing the number of filters at the start of the discriminator, the number of layers in the discriminator, and a list of boolean values indicating whether to include attention layers in the discriminator.
        g_config (tuple[int, int, list[bool]]): A tuple containing the number of filters at the start of the generator, the number of layers in the generator, and a list of boolean values indicating whether to include attention layers in the generator.

    Returns:
        WGAN: A compiled WGAN model with the specified discriminator and generator configurations.
    """
    d_filters_start, d_layer_count, d_att_layers_num = d_config
    g_filters_start, g_layer_count, g_att_layers_num = g_config

    d_model: Model = build_discriminator(
        img_size=img_size,
        filters_start=d_filters_start,
        layer_count=d_layer_count,
        norm=D_NORM.none,
        att_layers_num=d_att_layers_num,
    )

    g_model: Model = build_generator(
        img_size=img_size,
        latent_dim=latent_dim,
        filters_start=g_filters_start,
        layer_count=g_layer_count,
        norm=G_NORM.none,
        att_layers_num=g_att_layers_num,
    )

    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=latent_dim,
        discriminator_extra_steps=3,
    )

    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
        kid=KID(img_size, kid_image_size),
    )

    return wgan
