import keras_tuner
from keras.api.models import Model

from ganai.models import get_compiled_wgan
from ganai.utilites import DatasetFromDir


def tune(epochs: int, batch_size: int, model_config: dict) -> None:
    img_size = model_config["img_size"]
    kid_image_size = model_config["kid_image_size"]
    g_filters_multiplayer = model_config["g_filters_multiplayer"]
    g_attentions = model_config["g_attentions"]

    d_filters_multiplayer = model_config["d_filters_multiplayer"]
    d_attentions = model_config["d_attentions"]

    datasetLoader = DatasetFromDir("./data/anime/", img_size, batch_size, 1 / 50)

    train_data, val_data = datasetLoader()

    def build_model(hp: keras_tuner.HyperParameters) -> Model:
        noise_dim = hp.Int("noise_dim", min_value=8, max_value=512, step=8)
        g_filters_start = hp.Int("g_filters_start", min_value=4, max_value=64, step=4)
        d_filters_start = hp.Int("d_filters_start", min_value=4, max_value=64, step=4)

        return get_compiled_wgan(
            img_size=img_size,
            noise_dim=noise_dim,
            kid_image_size=kid_image_size,
            gen_config=(g_filters_start, g_filters_multiplayer, g_attentions),
            disc_config=(d_filters_start, d_filters_multiplayer, d_attentions),
        )

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective=keras_tuner.Objective("val_kid", "min"),
        max_trials=200,
        executions_per_trial=4,
        seed=2712,
        directory="./data/tuner",
        project_name="wgan",
        overwrite=True,
    )
    
    tuner.search(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        verbose=2,
    )

    tuner.results_summary()

    best_model = tuner.get_best_models()[0]
    best_model.summary()
