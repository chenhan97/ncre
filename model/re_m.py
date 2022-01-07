import torch
import torch.nn as nn
import transformers

gpu = torch.cuda.is_available()


class RelationExtractor(nn.Module):
    def __init__(self):
        super(RelationExtractor, self).__init__()
        bert_model, pretrained_weights = transformers.DistilBertModel, 'distilbert-base-uncased'
        self.bert_model = bert_model.from_pretrained(pretrained_weights)
        self.rel_attn = nn.Parameter(torch.randn(5, 768))
        self.rel_attn.requires_grad = True
        self.rel_classifer = nn.Linear(in_features=768, out_features=1)
        self.proj_layer = nn.Linear(in_features=768, out_features=1)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=0.4)
        self.A_r_matrix = nn.Linear(in_features=5, out_features=768)
        self.V_r_layer = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(512, 5), nn.Sigmoid())
        self.gate_layer = nn.Linear(in_features=1536, out_features=768)
        self.predict_layer = nn.Sequential(nn.Linear(2304, 512), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(512, 5))


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
            if gpu:
                tentative_ent_context = tentative_ent_context.cuda()
            for ent in ent_pos[i]:
                if ent[0] == ent[1]:
                    ent_context_emb = encoding[0][i][ent[0]]
                else:
                    ent_embs = encoding[0][i][ent[0]:ent[1] + 1]
                    ent_context_emb = torch.max_pool2d(ent_embs.reshape((1, 1, ent_embs.shape[0], ent_embs.shape[1])), kernel_size=(ent[1]-ent[0]+1, 1))
                tentative_ent_context = torch.cat((tentative_ent_context, ent_context_emb.reshape((1,-1))), dim=0)
            ent_context_emb = tentative_ent_context[1:] # num_ent * hidden_size
            ent_context_emb = self.dropout(ent_context_emb)
            init_first_ent_emb = ent_context_emb[0]
            init_second_ent_emb = ent_context_emb[1]
            init_third_ent_emb = ent_context_emb[2]

            for iter_ in range(5):
                update_first_ent_emb = torch.mul(self.A_r_matrix(self.V_r_layer(torch.cat((init_second_ent_emb, init_third_ent_emb), dim=0))), init_first_ent_emb)
                update_second_ent_emb = torch.mul(self.A_r_matrix(self.V_r_layer(torch.cat((init_first_ent_emb, init_third_ent_emb), dim=0))), init_second_ent_emb)
                update_third_ent_emb = torch.mul(self.A_r_matrix(self.V_r_layer(torch.cat((init_first_ent_emb, init_second_ent_emb), dim=0))), init_third_ent_emb)
                first_gate_score = torch.sigmoid(self.dropout(self.gate_layer(torch.cat((update_first_ent_emb, init_first_ent_emb)))))
                final_first_ent_emb = first_gate_score * init_first_ent_emb + (1-first_gate_score) * update_first_ent_emb
                second_gate_score = torch.sigmoid(self.dropout(self.gate_layer(torch.cat((update_second_ent_emb, init_second_ent_emb)))))
                final_second_ent_emb = second_gate_score * init_second_ent_emb + (1-second_gate_score) * update_second_ent_emb
                third_gate_score = torch.sigmoid(self.dropout(self.gate_layer(torch.cat((update_third_ent_emb, init_third_ent_emb)))))
                final_third_ent_emb = third_gate_score * init_third_ent_emb + (1-third_gate_score) * update_third_ent_emb
                init_first_ent_emb = final_first_ent_emb
                init_second_ent_emb = final_second_ent_emb
                init_third_ent_emb = final_third_ent_emb
            rel_result = self.predict_layer(torch.flatten(ent_context_emb)).unsqueeze(dim=0)
            rel_result_list.append(rel_result)

            '''
            attn_weights = torch.softmax(torch.matmul(ent_context_emb, self.rel_attn.T) / torch.sqrt(torch.tensor(768.0)), dim=0)  # num_ent * 5
            rel_attn_results = torch.matmul(ent_context_emb.T, attn_weights)  # hidden_size * 5
            rel_result = self.rel_classifer(rel_attn_results.T)  # 1*num_class
            rel_result_list.append(rel_result)'''

            ent_score = torch.sigmoid(self.proj_layer(ent_context_emb)).squeeze()
            ent_score_list.append(ent_score)
        return rel_result_list, ent_score_list
