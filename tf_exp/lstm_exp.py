import os
import numpy
import pickle
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from constants import SEP, UNK, PAD
from .metrics import Perplexity
from .layers import UnkPenaltyLayer
 
class Model:
    def __init__(self, vocab_file):
        vocab = [line.strip() for line in open(vocab_file)]
        self.w2id = {w:idx for idx, w in enumerate(vocab)}
        self.id2w = {idx:w for idx, w in enumerate(vocab)}
        self.id2w[self.w2id[SEP]] = "\n"
        self.unk_id = self.w2id[UNK]
        self.vocab_size = len(vocab)
        self.tokenizer = BertWordPieceTokenizer(vocab_file, sep_token=SEP)

    def build(self):
        inputs = tf.keras.Input(shape=(None,), )
        emb = tf.keras.layers.Embedding(self.vocab_size, 512)(inputs)
        emb = tf.keras.layers.Dropout(0.3)(emb)
        mid = tf.keras.layers.LSTM(4096, return_sequences=True, dropout=0.3)(emb)
        outputs = tf.keras.layers.Dense(self.vocab_size, activation=tf.math.softmax)(mid)
        outputs = UnkPenaltyLayer(unk_id = self.unk_id)(outputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
            metrics=[Perplexity(from_logits=False)]
        )
        self.model.summary()

    def train(self, x, y, batch_size=32, epochs=10):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def infer(self, text, generate_length=500):
        if not isinstance(text, str):
            raise Exception("*** inputs must be string ***")
        
        self.model.reset_states()

        outputs = []
        inputs = [self.w2id.get(w, self.unk_id) for w in text]
        for _ in range(generate_length):
            pred = tf.argmax(self.model.predict(numpy.array([inputs]))[0], axis=-1).numpy()[-1]
            outputs.append(self.id2w.get(pred, UNK))
            inputs.append(pred)
        return "".join(outputs)

    def save(self, model_path):
        self.model.save(model_path)
        return

    def load(self, model_path, custom_objects=None):
        self.model = tf.keras.models.load_model(model_path, custom_objects)
        return

    def show(self):
        self.model.summary()
        return

def main():
    print("=== loading dataset ===")
    src = pickle.load(open("data/src.pkl", "rb"))
    tgt = pickle.load(open("data/tgt.pkl", "rb"))

    print("=== initial model ===")
    model = Model("data/vocab.txt")
    model.build()

    print("=== training model ===")
    model.train(src, tgt, batch_size=128, epochs=5)

    print("=== saving model ===")
    model.save("model/tf_lstm_model_unk_penalty.h5")

    print("=== show infer result ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, generate_length=500)
    print(para)

def test():
    print("=== loading dataset ===")
    src = pickle.load(open("data/src.pkl", "rb"))[:1000]
    tgt = pickle.load(open("data/tgt.pkl", "rb"))[:1000]

    print("=== initial model ===")
    model = Model("data/vocab.txt")
    model.build()

    print("=== training model ===")
    model.train(src, tgt, batch_size=32, epochs=10)

    print("=== saving model ===")
    model.save("model/tf_lstm_model.h5")

    print("=== show infer result ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, generate_length=500)
    print(para)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # test()
    main()
