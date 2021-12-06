import torch
from utils import constant
import json
import transformers


class TrainDataLoader:

    def __init__(self, filePath, batch_size, cuda):
        inputs, ent_list, rel_list = self.read_file(filePath)
        self.inputs = [inputs[i:i + batch_size] for i in range(0, len(ent_list), batch_size)]
        self.ent_list = [ent_list[i:i + batch_size] for i in range(0, len(ent_list), batch_size)]
        self.rel_list = [rel_list[i:i + batch_size] for i in range(0, len(ent_list), batch_size)]
        self.num_batch = len(self.inputs)
        self.batch_counter = 0
        self.cuda = cuda

    def read_file(self, filePath):
        tokenizer_class, pretrained_weights = transformers.DistilBertTokenizer, 'distilbert-base-uncased'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        with open(filePath) as infile:
            data = json.load(infile)
        label2id = constant.LABEL_TO_ID
        sent_list = []
        ent_list = []
        rel_list = []
        for d_no, d in enumerate(data):
            sent_list.append(" ".join(d['token']))
            first_entity = [d['first_start'], d['first_end']]
            second_entity = [d['second_start'], d['second_end']]
            third_entity = [d['third_start'], d['third_end']]
            ent_list.append([first_entity, second_entity, third_entity])
            rel = label2id[d['relation']]
            rel_list.append(rel)
        inputs = tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs["input_ids"] # num_instances*seq_length(512)
        return inputs, ent_list, rel_list

    def next(self):
        batch_inputs = self.inputs[self.batch_counter]
        batch_ent = self.ent_list[self.batch_counter]
        batch_rel = self.rel_list[self.batch_counter]
        if self.batch_counter < self.num_batch - 1:
            self.batch_counter += 1
        else:
            self.batch_counter = 0
        return batch_inputs, batch_ent, batch_rel

    def get_num_batch(self):
        return self.num_batch
