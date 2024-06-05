import datetime

from keras.api.callbacks import ModelCheckpoint, CSVLogger

from ganai.configuration import Config
from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def train(
    epochs: int, batch_size: int, chp_path: str, verbose: int, config: Config
) -> None:
    img_size = config.img_size
    noise_dim = config.noise_dim

    kid_image_size = config.kid_image_size

    g_filters_start = config.g_filters_start
    g_filters_multiplayer = config.g_filters_multiplayer
    g_attentions = config.g_attentions

    d_filters_start = config.d_filters_start
    d_filters_multiplayer = config.d_filters_multiplayer
    d_attentions = config.d_attentions

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
            f'{chp_path}/model.weights.h5',
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
