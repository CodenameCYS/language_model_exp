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
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam()
        self.metrics = {"ppl": Perplexity(from_logits=False)}
        self.model.summary()

    @tf.function
    def train_one_step(self, x, y, batch_size):
        metrics = {x: 0 for x in self.metrics}
        with tf.GradientTape() as tape:
            loss = 0
            n = x.shape[0]
            batch_num = (n-1) // batch_size + 1
            for i in range(batch_num):
                y_pred = self.model(x[i*batch_size:(i+1)*batch_size])
                y_true = y[i*batch_size:(i+1)*batch_size]
                loss += self.loss_fn(y_true, y_pred) / n * y_true.shape[0]
                for metrics_name, metrics_fn in self.metrics.items():
                    metrics[metrics_name] += metrics_fn(y_true, y_pred) / n * y_true.shape[0]
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics
    
    def train(self, x, y, batch_size=32, epochs=10, effective_batch_size=None, checkpoint_path=None):
        if effective_batch_size is None:
            effective_batch_size = batch_size
        
        data_num = len(x)
        batch_num = (data_num-1) // effective_batch_size + 1
        for epoch in range(epochs):
            for i in range(batch_num):
                batch_x = x[i*effective_batch_size: (i+1)*effective_batch_size]
                batch_y = y[i*effective_batch_size: (i+1)*effective_batch_size]
                loss, metrics = self.train_one_step(batch_x, batch_y, batch_size)
                if i % 100 == 0:
                    print("epoch {}/{} , batch: {}/{} --> loss = {}".format(epoch+1, epochs, i, batch_num, loss), end="")
                    for metrics_name in self.metrics:
                        print(", {} = {}".format(metrics_name, metrics[metrics_name]), end="")
                    print()
            if epoch % 5 == 4 and checkpoint_path is not None:
                print("=== saving checkpoint ===")
                self.model.save_weights(os.path.join(checkpoint_path, "ckpt{}".format(epoch+1)))
        return

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

    print("=== show infer result before train ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, generate_length=500)
    print(para)

    print("=== training model ===")
    model.train(src, tgt, batch_size=128, epochs=5, effective_batch_size=128, checkpoint_path="model/tf_lstm_v2/")

    print("=== saving model ===")
    model.save("model/tf_lstm_model_v2.h5")

    print("=== show infer result after train ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, generate_length=500)
    print(para)

def test():
    print("=== loading dataset ===")
    src = pickle.load(open("data/src.pkl", "rb"))[:100]
    tgt = pickle.load(open("data/tgt.pkl", "rb"))[:100]

    print("=== initial model ===")
    model = Model("data/vocab.txt")
    model.build()

    print("=== show infer result before train ===")
    text = "只见林冲朝着那如花似玉，闭月羞花的林黛玉身边的红脸大汉大喝一声：“秃驴休走，吃俺老林一棒！”，那大汉轻浮长须，微微一笑，“酒且斟下，某去便来。”，说罢，"
    print("text:\n  {}\n...".format(text))
    para = model.infer(text, generate_length=500)
    print(para)

    print("=== training model ===")
    model.train(src, tgt, batch_size=32, epochs=7, effective_batch_size=64, checkpoint_path="model/tf_lstm_v2_tmp/")

    # print("=== saving model ===")
    # model.save("model/tf_lstm_model_v2_tmp.h5")

    print("=== show infer result after train ===")
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
