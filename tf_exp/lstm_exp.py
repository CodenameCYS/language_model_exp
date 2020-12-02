import os
import numpy
import pickle
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from constants import SEP, UNK, PAD
# from metrics import TensorflowPerplexity as Perplexity

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

    def result(self):
        return self.perplexity    

class Model:
    def __init__(self, vocab_file, window_size):
        vocab = [line.strip() for line in open(vocab_file)]
        self.w2id = {w:idx for idx, w in enumerate(vocab)}
        self.id2w = {idx:w for idx, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.window_size = window_size
        self.tokenizer = BertWordPieceTokenizer(vocab_file, sep_token=SEP)

    def build(self):
        inputs = tf.keras.Input(shape=(self.window_size,))
        emb = tf.keras.layers.Embedding(self.vocab_size, 512)(inputs)
        emb = tf.keras.layers.Dropout(0.3)(emb)
        mid = tf.keras.layers.LSTM(4096, return_sequences=True, dropout=0.3)(emb)
        outputs = tf.keras.layers.Dense(self.vocab_size, activation=tf.math.softmax)(mid)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
            metrics=[Perplexity(from_logits=False)]
        )
        
    def show_model(self):
        print("=== languange model structure ===")
        self.model.summary()

    def train(self, x, y, batch_size=32, epochs=10):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def infer(self, text, max_sentence_len=500):
        if isinstance(text, str):
            return self._infer([text], max_sentence_len)[0]
        elif isinstance(text, list) and all(isinstance(s, str) for s in text):
            return self._infer(text, max_sentence_len)
        else:
            raise Exception("*** inputs must be string or list of string! ***")

    def _infer(self, text_list, max_sentence_len=500):
        # self.model.reset_states()
        unk_id = self.w2id[UNK]
        tokens = [self.tokenizer.encode(text).tokens[1:-1] for text in text_list]
        tokens_id = [[self.w2id.get(w, unk_id) for w in s] for s in tokens]
        lengths = [len(s) for s in tokens]
        if any(s < self.window_size for s in lengths):
            print("inputs must have {} words, please input a longer begining!".format(self.window_size))        
        for i in range(max_sentence_len):
            inputs = numpy.array([s[-self.window_size:] for s in tokens_id])
            outputs = numpy.argmax(self.model.predict(inputs)[:, -1], axis=-1)
            for o, s, sid in zip(outputs, tokens, tokens_id):
                s.append(self.id2w.get(o, UNK))
                sid.append(o)
        return tokens

    def save(self, model_path):
        self.model.save(model_path)
        return

    def load(self, model_path, custom_objects=None):
        self.model = tf.keras.models.load_model(model_path, custom_objects)
        return

def main():
    print("=== loading dataset ===")
    src = pickle.load(open("data/src.pkl", "rb"))
    tgt = pickle.load(open("data/tgt.pkl", "rb"))

    print("=== initial model ===")
    model = Model("data/vocab.txt", window_size=len(src[0]))
    model.build()
    model.show_model()

    print("=== training model ===")
    model.train(src, tgt, batch_size=96, epochs=5)

    print("=== saving model ===")
    model.save("model/tf_lstm_model_v2.h5")

    print("=== show infer result ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, max_sentence_len=500)
    print(para)

def test():
    print("=== loading dataset ===")
    src = pickle.load(open("data/src.pkl", "rb"))[:100000]
    tgt = pickle.load(open("data/tgt.pkl", "rb"))[:100000]

    print("=== initial model ===")
    model = Model("data/vocab.txt", window_size=len(src[0]))
    model.show_model()

    print("=== training model ===")
    model.train(src, tgt, batch_size=32, epochs=10)

    print("=== saving model ===")
    model.save("model/tf_lstm_model.h5")

    print("=== show infer result ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, max_sentence_len=500)
    print(para)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # test()
    main()
