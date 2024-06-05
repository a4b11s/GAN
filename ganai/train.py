import datetime

from keras.api.callbacks import ModelCheckpoint, CSVLogger

from ganai.configuration import Config
from ganai.callbacks import GANMonitor
from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def start_train(epochs: int, config: Config) -> None:
    start_time = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

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
        "./data/anime/", img_size, batch_size, 1 / 200
    ).load_dataset()

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
    )

    loggerPath = f"./data/log/{model_id}-log.csv"

    train_callbacks: list[callable] = [
        GANMonitor(9),
        CSVLogger(loggerPath, append=True),
        ModelCheckpoint(f"./data/models/{model_id}-model.weights.h5", save_weights_only=True, verbose=1)
    ]
    
    wgan.summary()
    
    wgan.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=train_callbacks,
        verbose=1,
    )
