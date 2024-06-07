import tensorflow as tf

def generator_loss(fake_img: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_img)