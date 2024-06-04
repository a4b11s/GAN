import numpy as np

from ganai.models import get_compiled_wgan
from ganai.utilites import plot_image

from ganai.configuration import Config


from numpy.typing import NDArray


def generate_image(batch_size: int, batch_count: int, config: Config) -> None:
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

    output = wgan.generate(batch_size, batch_count)

    image_batchs: NDArray[np.float32] = output[0]
    seed: int = output[1]

    for index, batch in enumerate(image_batchs):
        plotted_image = plot_image((batch * 255).astype(np.uint8))

        plotted_image.save(f"generated/{index + 1}_seed-{seed}.png")
