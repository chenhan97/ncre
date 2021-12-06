import transformers
import torch
import numpy as np
import torch.nn as nn
from model.re_m import RelationExtractor
from utils import DataLoader
tokenizer_class, pretrained_weights = transformers.BertTokenizer, 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
x = ['a good place to stay .', 'where are you ?']
a = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
print(a)

input = a["input_ids"]
attention_mask = a["attention_mask"]
model = RelationExtractor()
model(input,[[[0,1],[2,3]],[[0,1],[2,3]]])

'''dl = DataLoader.TrainDataLoader("corpus/pubmed/test0.json", 12, False)
dl.next()
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
print(m(input), target)
output = loss(m(input), target)'''

