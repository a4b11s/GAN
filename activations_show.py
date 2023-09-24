import keract
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import img_to_array

from models.get_compiled_WGAN import get_compiled_wgan


def activations_show(config):
    img_size = config.img_size
    noise_dim = config.noise_dim

    kid_image_size = config.kid_image_size

    g_filters_start = config.g_filters_start
    g_filters_multiplayer = config.g_filters_multiplayer
    g_attentions = config.g_attentions

    d_filters_start = config.d_filters_start
    d_filters_multiplayer = config.d_filters_multiplayer
    d_attentions = config.d_attentions

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
    )

    model_id = 2
    checkpoint_path = f"checkpoints/{model_id}/model"

    wgan.load_weights(checkpoint_path)

    random_latent_vectors = tf.random.normal(shape=(1, noise_dim))

    image = Image.open('./datasets/ICC/ic/Ability_Rogue_SliceDice.png')
    image = image.convert('RGB')
    image = image.resize((img_size, img_size))
    image = [img_to_array(image)]
    arr_image = np.array(image)
    arr_image = arr_image / 255
    image = arr_image

    fake_img = wgan.generator.predict(random_latent_vectors)

    activations = keract.get_activations(wgan.generator, random_latent_vectors)
    activations_disc_real = keract.get_activations(wgan.discriminator, image)
    activations_disc_fake = keract.get_activations(wgan.discriminator, fake_img)

    keract.display_activations(activations, cmap="gray")
    keract.display_heatmaps(activations_disc_real, image, merge_filters=True)
    keract.display_heatmaps(activations_disc_fake, fake_img, merge_filters=True)
