import tensorflow as tf
from .utils import normalize

class UnkPenaltyLayer(tf.keras.layers.Layer):
    def __init__(self, unk_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unk_id = unk_id

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'unk_id': self.unk_id,
        })
        return config

    def call(self, inputs, training=False):
        if training:
            return inputs
        vocab_size = inputs.shape[-1]
        mask = 1 - tf.one_hot(self.unk_id * tf.reduce_max(tf.ones_like(inputs, dtype=tf.int64), axis=-1), vocab_size)
        return normalize(mask * inputs)