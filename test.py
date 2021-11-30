'''import transformers
import torch
import numpy as np
from model.re_m import RelationExtractor
tokenizer_class, pretrained_weights = transformers.DistilBertTokenizer, 'distilbert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
x = ['a good place to stay','. how about it ?']
for b in x:
    a = tokenizer.encode(b, add_special_tokens=True)
    print(a)
model = RelationExtractor()
a = [[[101, 1037, 2204, 2173, 2000, 2994, 102],[101, 1012, 2129, 2055, 2009, 1029, 102]], [[101, 1037, 2204, 2173, 2000, 2994, 0],[101, 1012, 2129, 2055, 2009, 1029, 0]]]
c = np.array(a)
b = np.where(c != 0, 1, 0)
model(a, b)
from utils import DataLoader

a = DataLoader.TrainDataLoader("corpus/pubmed/test0.json", 32,  False)
a.next()'''

from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper import JSON

queryString = """DESCRIBE wd:Q20145"""
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result)
