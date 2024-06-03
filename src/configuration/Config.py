class Config:
    def __init__(self):
        self.batch_size = 16

        self.img_size = 64
        self.noise_dim = 128

        self.kid_image_size = 75

        self.g_filters_start = 16
        self.g_filters_multiplayer = [8, 4, 2, 1]
        self.g_attentions = [False, False, False, True]

        self.d_filters_start = 32
        self.d_filters_multiplayer = [1, 2, 4, 8]
        self.d_attentions = [False, False, True, False]
