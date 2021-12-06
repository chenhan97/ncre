import torch
import torch.nn as nn
import transformers

gpu = torch.cuda.is_available()


class RelationExtractor(nn.Module):
    def __init__(self):
        super(RelationExtractor, self).__init__()
        bert_model, pretrained_weights = transformers.DistilBertModel, 'distilbert-base-uncased'
        self.bert_model = bert_model.from_pretrained(pretrained_weights)
        self.rel_attn = nn.Parameter(torch.randn(1, 768))
        self.rel_attn.requires_grad = True
        self.rel_classifer = nn.Linear(in_features=768, out_features=5)
        self.proj_layer = nn.Linear(in_features=768, out_features=1)

    def forward(self, inputs, ent_pos):
        """
        a pre-trained BERT is used to learn the embedding of each sentence. relation is learned by attending the
        embedding of each entity. Entities are scored with a logistic layer.
        :param inputs: a list of tokens with the
        BERT special token added. format: [[[idx1, idx2...],[idx1,...]], [[idx1..],[idx3..]]] (
        batch*words)
        :param ent_pos: a list of the positions of entities in each sentence group.
        format: [batch_size*num_ent] e.g., [[0,1],[2,5,6]]
        :return: a list of relation classification results.[cls1,
        cls2, ...]
        TODO: batch process in some computing stages.
        """
        encoding = self.bert_model(inputs, return_dict=False)
        # encoding[0] is the hidden state: batch_size*seq_length*hidden_size
        ent_score_list = []
        rel_result_list = []
        for i in range(encoding[0].shape[0]):
            tentative_ent_context = torch.zeros((1, encoding[0].shape[2]))
            for ent in ent_pos[i]:
                if ent[0] == ent[1]:
                    ent_context_emb = encoding[0][i][ent[0]]
                else:
                    ent_embs = encoding[0][i][ent[0]:ent[1] + 1]
                    ent_context_emb = torch.max_pool2d(ent_embs.reshape((1, 1, ent_embs.shape[0], ent_embs.shape[1])), kernel_size=(ent[1]-ent[0]+1, 1))
                tentative_ent_context = torch.cat((tentative_ent_context, ent_context_emb.reshape((1,-1))), dim=0)
            ent_context_emb = tentative_ent_context[1:] # num_ent * hidden_size
            attn_weights = torch.softmax(torch.matmul(ent_context_emb, self.rel_attn.T) / torch.sqrt(torch.tensor(768)), dim=0)  # num_ent * 1
            rel_attn_results = torch.matmul(ent_context_emb.T, attn_weights)  # hidden_size * 1
            rel_result = torch.softmax(self.rel_classifer(rel_attn_results.T), dim=1)  # 1*num_class
            rel_result_list.append(rel_result)
            ent_score = torch.sigmoid(self.proj_layer(ent_context_emb)).squeeze()
            ent_score_list.append(ent_score)
        return rel_result_list, ent_score_list
