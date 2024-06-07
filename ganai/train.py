from keras.api.callbacks import ModelCheckpoint

from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def train(
    epochs: int, batch_size: int, chp_path: str, verbose: bool, model_config: dict
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

    datasetLoader = DatasetFromDir("./data/anime/", img_size, batch_size, 1 / 200)

    train_data, val_data = datasetLoader()

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
    )

    chp_file = f"{chp_path}/model.weights.h5"

    train_callbacks: list[callable] = [
        GANMonitor(9),
        ModelCheckpoint(
            chp_file,
            save_weights_only=True,
            verbose=1 if verbose else 0,
        ),
    ]

    try:
        wgan.load_weights(chp_file)
        if verbose:
            print(f"Model weights loaded from {chp_file}")
    except:
        if verbose:
            print(f"Model weights not loaded from {chp_file}")

    if verbose:
        wgan.summary()

    wgan.fit(
        train_data,
        epochs=epochs,
        callbacks=train_callbacks,
        verbose=1 if verbose else 2,
    )
