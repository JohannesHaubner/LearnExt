import torch
import torch.nn as nn
from torch_extension.networks import MLP, Normalizer



def load_yankee():

    x_mean = torch.tensor([ 8.4319e-01,  2.0462e-01, -1.5600e-03,  4.7358e-04, -5.4384e-03,
            9.0626e-04,  1.3179e-03,  7.8762e-04])
    x_std = torch.tensor([0.6965, 0.1011, 0.0035, 0.0134, 0.0425, 0.0468, 0.1392, 0.1484])
    normalizer = Normalizer(x_mean, x_std)

    widths = [8] + [128]*6 + [2]
    mlp = MLP(widths, activation=nn.ReLU())
    net = nn.Sequential(normalizer, mlp)
    net.load_state_dict(torch.load("torch_extension/models/yankee/state_dict.pt"))
#     net.double()
    net.eval()

    return net
