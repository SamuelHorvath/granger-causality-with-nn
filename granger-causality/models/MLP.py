import torch
import torch.nn as nn

from .utils import TrainableEltWiseLayer, activation_helper


class CMLP(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1, act='relu'):
        super(CMLP, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size

        # Linear Layer Input
        self.linear = nn.Linear(input_size * seq_len, hidden_dim)

        # Linear layers
        self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Linear Layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # activation
        self.act = activation_helper(act)

    def forward(self, input):
        out = self.act(self.linear(input))
        for layer in self.hidden_s:
            out = self.act(layer(out))
        out = self.fc(out)
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time


class CMLPFull(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='relu'):
        super(CMLPFull, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.networks = nn.ModuleList([
            CMLP(input_size=input_size, seq_len=seq_len, hidden_dim=hidden_dim,
                 n_layers=n_layers, act=act)
            for _ in range(input_size)])

    def forward(self, input):
        input_flatten = input.reshape(-1, self.seq_len * self.input_size)
        out = torch.cat([network(input_flatten) for network in self.networks], dim=1)
        return out


class CMLPwFilter(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1, act='relu'):
        super(CMLPwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size

        # Filter
        self.filter = TrainableEltWiseLayer(input_size * seq_len)
        # Linear Layer Input
        self.linear = nn.Linear(input_size * seq_len, hidden_dim)

        # LSTM layers
        self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Linear Layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # activation
        self.act = activation_helper(act)

    def forward(self, input):
        filtered_input = self.filter(input)
        out = self.act(self.linear(filtered_input))
        for layer in self.hidden_s:
            out = self.act(layer(out))
        out = self.fc(out)
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time


class CMLPwFilterFull(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='relu'):
        super(CMLPwFilterFull, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.networks = nn.ModuleList([
            CMLPwFilter(input_size=input_size, seq_len=seq_len, hidden_dim=hidden_dim,
                        n_layers=n_layers, act=act)
            for _ in range(input_size)])

    def forward(self, input):
        input_flatten = input.reshape(-1, self.seq_len * self.input_size)
        out = torch.cat([network(input_flatten) for network in self.networks], dim=1)
        return out


class LeKVAR(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='relu'):
        super(LeKVAR, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Linear Layer Input
        self.linear = nn.Linear(input_size, hidden_dim)

        # LSTM layers
        self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Linear Layer ReLU
        self.fc = nn.Linear(hidden_dim * seq_len, input_size)

        # activation
        self.act = activation_helper(act)

    def forward(self, input):
        out = self.linear(input)
        for layer in self.hidden_s:
            out = self.act(out)
            out = layer(out)
        out = self.fc(out.reshape(-1, self.seq_len * self.hidden_dim))
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.fc.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.fc.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time


def cmlp(input_size=10, seq_len=10):
    return CMLPFull(input_size, seq_len)


def cmlpwf(input_size=10, seq_len=10):
    return CMLPwFilterFull(input_size, seq_len)


def lekvar(input_size=10, seq_len=10):
    return LeKVAR(input_size, seq_len)
