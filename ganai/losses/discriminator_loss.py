import tensorflow as tf


def discriminator_loss(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss
