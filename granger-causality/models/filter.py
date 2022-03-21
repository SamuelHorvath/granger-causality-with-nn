import torch
from torch import nn


class TrainableEltWiseLayer(nn.Module):

    def __init__(self, n):
        super(TrainableEltWiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(n))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weights.data, 0.)

    def forward(self, x):
        return x * torch.sigmoid(self.weights)
