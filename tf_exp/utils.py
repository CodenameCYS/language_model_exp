import tensorflow as tf

def normalize(inputs):
    shape = inputs.shape
    vocab_size = shape[-1]
    tile_shape = [1 for x in shape[:-1]] + [vocab_size]
    _sum = tf.tile(tf.reduce_sum(inputs, axis=-1, keepdims=True), tile_shape)
    return inputs / _sum
