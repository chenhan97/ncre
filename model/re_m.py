import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

gpu = torch.cuda.is_available()


class RelationExtractor(nn.Module):
    def __init__(self):
        super(RelationExtractor, self).__init__()
        bert_model, pretrained_weights = transformers.DistilBertModel, 'distilbert-base-uncased'
        self.bert_model = bert_model.from_pretrained(pretrained_weights)
        self.info_transform_layer = nn.LSTMCell(input_size=768, hidden_size=128)
        self.classifer = nn.Linear(in_features=128, out_features=1)

    def forward(self, inputs, masks):
        """
        a pre-trained BERT is used to learn the embedding of each sentence. A LSTM layer is considered as a
        information transformation layer that indicates the relation information construction in each sentence.
        problems to investigate: 1. Is that LSTM layer better than BERT in terms of modeling the whole text? 2. Is
        there a better way to replace the LSTM layer?
        :param inputs: a list of tokens with the BERT special token
        added. format: [[[idx1, idx2...],[idx1,...]], [[idx1..],[idx3..]]] (batch*num_sent*num_words)
        :param masks: a list of 2-D narry that represents the real tokens and paddings. format: [num_sent*num_words, num_sent*num_words...]
        :return: a list of relation classification results.[cls1, cls2, ...]
        """
        classifications = []
        for i in range(len(inputs)):
            tensor_input = torch.Tensor(inputs[i]).long()
            with torch.no_grad():
                last_hidden_states = self.bert_model(tensor_input, attention_mask=torch.Tensor(masks[i]))
                # features dimension: num_sent*768
                features = last_hidden_states[0][:, 0, :]
            for j in range(len(features)):
                if j == 0:
                    h, c = self.info_transform_layer(features[j].unsqueeze(dim=0))
                    prev_h, prev_c = h, c
                else:
                    h, c = self.info_transform_layer(features[j].unsqueeze(dim=0), (prev_h, prev_c))
                    prev_h, prev_c = h, c
            classification = self.classifer(h)
            classifications.append(classification)
        return classifications
