import math
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
from keras import Model
from numpy.typing import NDArray

from ganai.utilites import KID


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator: Model,
        generator: Model,
        latent_dim: int,
        discriminator_extra_steps: int = 3,
        gp_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        self.kid: KID = None
        self.g_loss_fn = None
        self.d_loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, kid) -> None:
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.kid = kid

    def gradient_penalty(self, batch_size, real_images, fake_images) -> tf.Tensor:
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images) -> dict[str, tf.Tensor]:
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake output from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake output
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real output
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake output using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake output
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def test_step(self, real_images) -> dict[str, tf.Tensor]:
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)

        self.kid.update_state(real_images, generated_images)

        # only KID is measured during the evaluation phase for computational efficiency
        return {self.kid.name: self.kid.result()}

    def generate(
        self, batch_size: int = 1, batch_count: int = 1, seed: int = None
    ) -> tuple[NDArray[np.float32], int]:
        if seed is None:
            seed = math.floor(time.time())
        tf.random.set_seed(seed)
        random_latent_vectors = tf.random.normal(
            shape=(batch_count, batch_size, self.latent_dim)
        )

        generated = []

        for batch in random_latent_vectors:
            generated_images = self.generator(batch)
            generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)
            generated.append(generated_images)

        generated = np.array(generated, dtype=np.float32)

        return generated, seed

    def plot_images(
        self,
        epoch=None,
        logs=None,
        save_dir="output",
        num_rows=2,
        num_cols=2,
        seed=None,
        is_show=False,
    ) -> None:

        if seed is None:
            seed = math.floor(time.time())

        tf.random.set_seed(seed)

        batch_size = num_rows * num_cols

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)

        if seed == 2712:
            generated_images = generated_images * 0.5 + 0.5

        plt.axis("off")
        plt.tight_layout()
        fig, ax = plt.subplots(num_rows, num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].axis("off")
                ax[i, j].set_aspect("equal")
                ax[i, j].imshow(generated_images[i + j])
        if is_show:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/e{'' if epoch is None else epoch + 1}_s{seed}.png")

        plt.close(fig)
