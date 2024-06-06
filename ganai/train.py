from keras.api.callbacks import ModelCheckpoint

from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def train(
    epochs: int, batch_size: int, chp_path: str, verbose: int, model_config: dict
) -> None:

    img_size = model_config["img_size"]
    noise_dim = model_config["noise_dim"]
    kid_image_size = model_config["kid_image_size"]
    g_filters_start = model_config["g_filters_start"]
    g_filters_multiplayer = model_config["g_filters_multiplayer"]
    g_attentions = model_config["g_attentions"]

    d_filters_start = model_config["d_filters_start"]
    d_filters_multiplayer = model_config["d_filters_multiplayer"]
    d_attentions = model_config["d_attentions"]


    train_data, val_data = DatasetFromDir(
        "./data/anime/", img_size, batch_size, 1 / 200
    ).load_dataset()

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
    )

    train_callbacks: list[callable] = [
        GANMonitor(9),
        ModelCheckpoint(
            f"{chp_path}/model.weights.h5",
            save_weights_only=True,
            verbose=0 if verbose else 0,
        ),
    ]

    wgan.summary()

    wgan.fit(
        train_data,
        epochs=epochs,
        callbacks=train_callbacks,
        verbose=1,
    )
