import torch
import tensorflow as tf

class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.perplexity = self.add_weight(name="ppl", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        vocab_size = y_pred.shape[-1]
        y_true = tf.one_hot(y_true, vocab_size)
        p = tf.math.reduce_sum(y_true * y_pred, axis=-1)
        ppl = tf.math.pow(1/tf.math.reduce_prod(p, axis=-1), 1/p.shape[-1])
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, ppl.shape)
            ppl = tf.multiply(ppl, sample_weight)
        self.perplexity.assign(tf.math.reduce_mean(ppl))
        return

    def result(self):
        return self.perplexity  

class TorchPerplexity:
    def __init__(self, from_logits=True):
        self.from_logits = from_logits

    def __call__(self, y_pred, y_true):
        pass