import torch
from utils import constant
import json
import random
import transformers
import nltk


class TrainDataLoader:

    def __init__(self, filePath, batch_size, cuda):
        # self.word2id, _ = build_vocab(filePath)
        data = self.read_file(filePath)
        self.data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.num_batch = len(self.data)
        self.batch_counter = 0
        self.cuda = cuda

    def read_file(self, filePath):
        tokenizer_class, pretrained_weights = transformers.DistilBertTokenizer, 'distilbert-base-uncased'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        with open(filePath) as infile:
            data = json.load(infile)
        label2id = constant.LABEL_TO_ID
        processed = []
        for d_no, d in enumerate(data):
            tokens = list(d['token'])
            sent_text = nltk.sent_tokenize(" ".join(tokens))
            instance = []
            for sent in sent_text:
                idx_sent = tokenizer.encode(sent, add_special_tokens=True)
                instance.append(idx_sent)
            relation = label2id[d['relation']]
            processed.append([instance, relation])
        indices = list(range(len(processed)))
        random.shuffle(indices)
        processed = [processed[i] for i in indices]
        return processed

    def next(self):
        batch = self.data[self.batch_counter]
        if self.batch_counter < self.num_batch - 1:
            self.batch_counter += 1
        else:
            self.batch_counter = 0
        batch_size = len(batch)
        batch = list(zip(*batch))
        # sort all fields by lens for easy RNN operations
        rels = batch[1]
        inputs = batch[0]
        return inputs, rels

    def get_num_ent(self):
        return len(self.ent2id)

    def get_num_batch(self):
        return self.num_batch
