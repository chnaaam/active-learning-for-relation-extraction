import torch.nn.functional as F


def cross_entropy_loss(true_y, pred_y):
    return F.cross_entropy(pred_y, true_y)