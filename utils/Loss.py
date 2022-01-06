import torch
import torch.nn as nn

gpu = torch.cuda.is_available()

def rel_classify_loss(rel_list_pred, rel_list_real):
    """
    calculate the relation classification loss using crossentropy loss in each batch
    :param rel_list_pred: A list of predicted tensors. Format:[tensor[1*num_class], tensor[1*num_class]...]
    :param rel_list_real: A list of relation ids. Format:[r_idx1, r_idx2, ...]
    :return: the loss of relation classification
    """
    rel_pred = torch.stack(rel_list_pred).squeeze()
    rel_real = torch.Tensor(rel_list_real).long()
    loss_function = nn.CrossEntropyLoss()
    if gpu:
        loss_function = loss_function.cuda()
        rel_real = rel_real.cuda()
    rel_class_loss = loss_function(rel_pred, rel_real)
    return rel_class_loss


def ent_detect_loss(ent_score_list, real_ent_list):
    """
    calculate the entity classification loss using BCEloss in each batch
    :param ent_score_list: A list of tensor scores of each entity. Format:[tensor[score_ent1, score_ent2],tensor[score_ent1, score_ent2]...]
    :param real_ent_list: A list of 1-0 indicators of participated entities. Format:[[1,0],[0,1],...]
    :return: the loss of entity classification
    """
    ent_score_pred = torch.cat(ent_score_list)
    ent_indicators = torch.Tensor([indicator for batch in real_ent_list for indicator in batch])
    loss_function = nn.BCELoss()
    if gpu:
        loss_function = loss_function.cuda()
        ent_indicators = ent_indicators.cuda()
    ent_class_loss = loss_function(ent_score_pred, ent_indicators)
    return ent_class_loss
