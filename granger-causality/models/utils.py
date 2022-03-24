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


def activation_helper(activation):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act
