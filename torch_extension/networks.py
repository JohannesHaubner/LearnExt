import torch
import torch.nn as nn

from typing import Callable

class MLP(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."

        return self.layers(x)

class Normalizer(nn.Module):
        def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor):
            super().__init__()
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
            self.x_std = nn.Parameter(x_std, requires_grad=False)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (x - self.x_mean) / self.x_std

class MLP_BN(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation
        self.bn = nn.BatchNorm1d(3935)

        layers = []
        layers = [nn.BatchNorm1d(3935)]
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."

        return self.layers(x)
