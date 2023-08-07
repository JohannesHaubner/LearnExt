import torch
import torch.nn as nn
from torch_extension.networks import MLP_BN


def load_sierra():

    widths = [8] + [64]*10 + [2]
    mlp = MLP_BN(widths, activation=nn.ReLU())
    mlp.load_state_dict(torch.load("torch_extension/model/sierra/state_dict.pt"))
    mlp.double()
    mlp.eval()
    return mlp