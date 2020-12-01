import os
import re
import numpy
import pickle
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer

from constants import SEP, UNK, PAD

# 分词器我们采用bert的分词器，其实现我们直接采用huggingface的tokenizers开源工具。
vocab_file = "data/vocab.txt"
tokenizer = BertWordPieceTokenizer(vocab_file, sep_token=SEP)

class DataProcessor:
    def __init__(self, corpus, vocab_file, tokenizer = tokenizer):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.w2id = {line.strip():idx for idx, line in enumerate(open(vocab_file))}

    def _preprocess(self, datafile, window_size):
        src_list = []
        tgt_list = []
        text = re.sub("\n+", "[ENTER]", open(datafile).read().strip())
        print("=== loading file ===")
        tokens = self.tokenizer.encode(text).tokens[1:-1]
        n = len(tokens)
        if n <= window_size:
            raise Exception("*** please check datafile, text has less than {} words! ***".format(window_size+1))
        unk_id = self.w2id[UNK]
        with tqdm(range(n-window_size), ncols=100) as t:
            for i in t:
                src = [self.w2id.get(w, unk_id) for w in tokens[i:i+window_size]]
                src_list.append(src)
                tgt = [self.w2id.get(w, unk_id) for w in tokens[i+1:i+window_size+1]]
                tgt_list.append(tgt)
                # tgt_list.append(self.w2id.get(tokens[i+window_size], unk_id))
        return src_list, tgt_list

    def preprocess_data(self, foutpath, window_size=64):
        src_list = []
        tgt_list = []
        for datafile in os.listdir(self.corpus):
            print("=== {} ===".format(datafile))
            datafile = os.path.join(self.corpus, datafile)
            src, tgt = self._preprocess(datafile, window_size)
            src_list.extend(src)
            tgt_list.extend(tgt)
        print("=== saving data ===")
        pickle.dump(numpy.array(src_list), open(os.path.join(foutpath, "src.pkl"), "wb"))
        pickle.dump(numpy.array(tgt_list), open(os.path.join(foutpath, "tgt.pkl"), "wb"))
        return

if __name__ == "__main__":
    corpus = "data/corpus"
    data_processor = DataProcessor(corpus, vocab_file, tokenizer)
    data_processor.preprocess_data("data/", window_size=48)