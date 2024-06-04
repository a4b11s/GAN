from keras.api.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.framework.errors_impl import NotFoundError

from ganai.configuration import Config
from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def start_train(epochs: int, config: Config) -> None:
    model_id = 1

    batch_size = config.batch_size

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
        "/data/anime/", img_size, batch_size, 1 / 10
    ).load_dataset()

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
        is_summary=True,
    )

    checkpoint_path = f"/data/checkpoints/{model_id}/model"
    loggerPath = f"/data/log/{model_id}-log.csv"

    try:
        wgan.load_weights(checkpoint_path)
        print("Model checkpoint loaded")
        wgan.plot_images(epoch=-1)
    except NotFoundError:
        print("Model checkpoint not found")

    train_callbacks: list[callable] = [
        ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
        GANMonitor(9),
        CSVLogger(loggerPath, append=True),
    ]

    wgan.fit(
        train_data,
        validation_data=val_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=train_callbacks,
    )
