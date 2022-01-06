import torch
from utils import DataLoader, Loss
from model import re_m


def train(args):
    num_iter = args.num_iter
    # initilize the model
    re_model = re_m.RelationExtractor()
    optimizer = torch.optim.Adam(re_model.parameters(), lr=0.001)
    if args.gpu:
        re_model = re_model.cuda()

    # process data
    dataloader = DataLoader.TrainDataLoader(args.train_data, args.batch_size, args.gpu)

    # training loop
    for i in range(num_iter):
        avg_loss = 0.0
        avg_rel_loss = 0.0
        avg_ent_loss = 0.0
        num_batch = dataloader.get_num_batch()
        for j in range(num_batch):
            optimizer.zero_grad()
            batch_inputs, batch_ent, batch_rel, batch_indicator = dataloader.next()
            rel_result_list, ent_score_list = re_model(batch_inputs, batch_ent)
            rel_loss = Loss.rel_classify_loss(rel_result_list, batch_rel)
            ent_loss = Loss.ent_detect_loss(ent_score_list, batch_indicator)
            avg_rel_loss += rel_loss.item()
            avg_ent_loss += ent_loss.item()
            loss = rel_loss + ent_loss
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        if num_iter % 10 == 0:
            torch.save(re_model.state_dict(), args.save_model_dir + "/check_point" + str(i))
        print("Iter: ", i, " Avg Loss: ", avg_loss / num_batch)
        print("Iter: ", i, " Avg Rel Loss: ", avg_rel_loss / num_batch)
        print("Iter: ", i, " Avg Ent Loss: ", avg_ent_loss / num_batch)
