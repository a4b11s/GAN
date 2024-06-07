import keras_tuner
from keras.api.models import Model
from keras.api.callbacks import EarlyStopping

from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def tune(epochs: int, batch_size: int, model_config: dict) -> None:
    img_size = model_config["img_size"]
    latent_dim = model_config["latent_dim"]
    kid_image_size = model_config["kid_image_size"]
    g_layer_count = model_config["g_layer_count"]
    g_att_layers_num = model_config["g_att_layers_num"]

    d_layer_count = model_config["d_layer_count"]
    d_att_layers_num = model_config["d_att_layers_num"]

    datasetLoader = DatasetFromDir("./data/anime/", img_size, batch_size, 1 / 50)

    train_data, val_data = datasetLoader()
    train_data = train_data.take(5000)
    val_data = val_data.take(500)

    def build_model(hp: keras_tuner.HyperParameters) -> Model:
        g_filters_start = hp.Int("g_filters_start", min_value=4, max_value=128, step=4)
        d_filters_start = hp.Int("d_filters_start", min_value=4, max_value=128, step=4)

        return get_compiled_wgan(
            img_size=img_size,
            latent_dim=latent_dim,
            kid_image_size=kid_image_size,
            g_config=(g_filters_start, g_layer_count, g_att_layers_num),
            d_config=(d_filters_start, d_layer_count, d_att_layers_num),
        )

    tuner = keras_tuner.Hyperband(
        build_model,
        objective=keras_tuner.Objective("val_kid", "min"),
        max_epochs=epochs,
        hyperband_iterations=2,
        directory="./data/tuner",
        project_name="wgan",
        overwrite=False,
    )

    stop_early = EarlyStopping(
        monitor="val_kid", min_delta=0.001, patience=4, mode="min"
    )

    tuner.search(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[stop_early],
        verbose=1,
    )

    tuner.results_summary()

    best_model = tuner.get_best_models()[0]
    best_model.summary()
