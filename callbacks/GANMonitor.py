from keras.callbacks import Callback


class GANMonitor(Callback):
    def __init__(self, num_rows=3, num_cols=3, is_random_seed=None, fixed_seed=None):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.is_random_seed = is_random_seed
        self.fixed_seed = fixed_seed

        if fixed_seed is None and is_random_seed is None:
            self.is_random_seed = True

    def on_epoch_end(self, epoch, logs=None):
        if self.is_random_seed:
            self.model.plot_images(epoch, num_rows=self.num_rows, num_cols=self.num_cols)

        if self.fixed_seed is not None:
            self.model.plot_images(epoch, num_rows=self.num_rows, num_cols=self.num_cols, seed=self.fixed_seed)
