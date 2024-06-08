import tensorflow as tf
from keras.api.callbacks import ModelCheckpoint

from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def train(
    epochs: int,
    batch_size: int,
    chp_path: str,
    verbose: bool,
    model_config: dict,
    is_mv: bool = False,
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

    datasetLoader = DatasetFromDir("./data/anime/", img_size, batch_size, 1 / 200)

    if is_mv:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            train_data, val_data = datasetLoader()

            wgan = get_compiled_wgan(
                img_size=img_size,
                latent_dim=latent_dim,
                kid_image_size=kid_image_size,
                g_config=(g_filters_start, g_layer_count, g_att_layers_num),
                d_config=(d_filters_start, d_layer_count, d_att_layers_num),
            )
    else:
        train_data, val_data = datasetLoader()

        wgan = get_compiled_wgan(
            img_size=img_size,
            latent_dim=latent_dim,
            kid_image_size=kid_image_size,
            g_config=(g_filters_start, g_layer_count, g_att_layers_num),
            d_config=(d_filters_start, d_layer_count, d_att_layers_num),
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
        validation_data=val_data,
        callbacks=train_callbacks,
        verbose=1 if verbose else 2,
    )
