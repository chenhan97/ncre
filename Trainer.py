import torch
from utils import DataLoader, Loss
from model import re_m


def train():
    num_iter = 10
    # initilize the model
    re_model = re_m.RelationExtractor()
    optimizer = torch.optim.Adam(re_model.parameters(), lr=0.001)

    # process data
    dataloader = DataLoader.TrainDataLoader("corpus/pubmed/test0.json", 12, False)

    # training loop
    for i in range(num_iter):
        for j in range(dataloader.get_num_batch()):
            optimizer.zero_grad()
            batch_inputs, batch_ent, batch_rel = dataloader.next()
            rel_result_list, ent_score_list = re_model(batch_inputs, batch_ent)
            rel_loss = Loss.rel_classify_loss(rel_result_list, batch_rel)
            ent_loss = Loss.ent_detect_loss(ent_score_list, None)
            loss = rel_loss + ent_loss
            loss.backward()
            optimizer.step()



