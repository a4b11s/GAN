import numpy as np

from ganai.models import get_compiled_wgan
from ganai.utilites import ImagePlotter

from numpy.typing import NDArray


def generate_image(
    chp_path: str,
    output_path: str,
    batch_size: int,
    batch_count: int,
    verbose: bool,
    model_config: dict,
) -> None:
    img_size = model_config["img_size"]
    latent_dim = model_config["latent_dim"]
    kid_image_size = model_config["kid_image_size"]
    g_filters_start = model_config["g_filters_start"]
    g_layer_count = model_config["g_layer_count"]
    g_att_layers_num = model_config["g_att_layers_num"]

    d_filters_start = model_config["d_filters_start"]
    d_layer_count = model_config["d_layer_count"]
    d_att_layers_num = model_config["d_att_layers_num"]

    wgan = get_compiled_wgan(
        img_size=img_size,
        latent_dim=latent_dim,
        kid_image_size=kid_image_size,
        g_config=(g_filters_start, g_layer_count, g_att_layers_num),
        d_config=(d_filters_start, d_layer_count, d_att_layers_num),
    )

    wgan.load_weights(chp_path)

    output = wgan.generate(batch_size, batch_count)

    image_batchs: NDArray[np.float32] = output[0]
    seed: int = output[1]

    plotter = ImagePlotter()

    for index, batch in enumerate(image_batchs):
        batch = (batch * 255).astype(np.uint8)

        plotter(batch, save_path=f"{output_path}/{index + 1}_seed-{seed}.png")
