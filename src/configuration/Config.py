class Config:
    def __init__(self) -> None:
        self.batch_size: int = 16

        self.img_size: int = 64
        self.noise_dim: int = 128

        self.kid_image_size: int = 75

        self.g_filters_start: int = 16
        self.g_filters_multiplayer: list[int] = [8, 4, 2, 1]
        self.g_attentions: list[bool] = [False, False, False, True]

        self.d_filters_start: int = 32
        self.d_filters_multiplayer: list[int] = [1, 2, 4, 8]
        self.d_attentions: list[bool] = [False, False, True, False]
