import tensorflow as tf
from keras.optimizers import Adam

from architectures.self_attention_discriminator import build_discriminator
from architectures.self_attention_generator import build_sa_generator
from models.WGAN import WGAN
from utilites.KID import KID


def get_compiled_wgan(img_size, noise_dim, kid_image_size, disc_config, gen_config, is_summary=False):
    d_filters_start, d_filters_multiplayer, d_attentions = disc_config
    g_filters_start, g_filters_multiplayer, g_attentions = gen_config

    d_model = build_discriminator(img_size, d_filters_start, d_filters_multiplayer, d_attentions)
    g_model = build_sa_generator(noise_dim, g_filters_start, g_filters_multiplayer, g_attentions)

    if is_summary:
        d_model.summary()
        g_model.summary()

    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
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
