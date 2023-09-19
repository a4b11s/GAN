from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.framework.errors_impl import NotFoundError

from callbacks.GANMonitor import GANMonitor
from models.get_compiled_WGAN import get_compiled_wgan
from utilites.dataset_generator import LoadItem


def start_train(epochs: int = 10):
    model_id = 6

    batch_size = 20

    img_size = 64
    noise_dim = 64

    kid_image_size = 75

    g_filters_start = 32
    g_filters_multiplayer = [8, 4, 2, 1]
    g_attentions = [False, False, False, True]

    d_filters_start = 32
    d_filters_multiplayer = [1, 2, 4, 8]
    d_attentions = [True, False, False, True]

    train_data = LoadItem("datasets/ICC/", img_size, batch_size, "one")

    wgan = get_compiled_wgan(
        img_size=img_size,
        noise_dim=noise_dim,
        kid_image_size=kid_image_size,
        gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
        disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
        is_summary=True,
    )

    checkpoint_path = f"checkpoints/{model_id}/model"
    loggerPath = f"log/{model_id}-log.csv"

    try:
        wgan.load_weights(checkpoint_path)
        print("Model checkpoint loaded")
        wgan.plot_images(epoch=-1)
    except NotFoundError:
        print("Model checkpoint not found")

    train_callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
        GANMonitor(num_rows=3, num_cols=3),
        CSVLogger(loggerPath, append=True)
    ]

    wgan.fit(train_data, batch_size=batch_size, epochs=epochs,
             callbacks=train_callbacks)
