import torch
from torch import Tensor, nn
def softmax(x, i):
    shifted_x = x - torch.amax(x, dim=i, keepdim=True)
    exp_x = torch.exp(shifted_x)
    return exp_x / torch.sum(exp_x, dim=i, keepdim=True)