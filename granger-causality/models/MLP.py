import torch
import torch.nn as nn
import numpy as np

from torch.linalg import norm

from .utils import TrainableEltWiseLayer, activation_helper


class CMLP(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1, act='sigmoid'):
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

    def regularize(self, lam):
        # Group Lasso on features
        reg_features = lam * norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,)).sum()
        # Group Lasso on time steps
        reg_lags = lam * norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,)).sum()
        return reg_features + reg_lags

    @torch.no_grad()
    def prox(self, lam):
        # Group Lasso on features
        shape = self.linear.weight.shape
        self.linear.weight.data = (nn.ReLU()(
            1 - lam / norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,), keepdim=True)) *
            self.linear.weight.reshape(-1, self.input_size)).reshape(shape)
        # Group Lasso on time steps
        shape_t = self.linear.weight.T.shape
        self.linear.weight.data = (nn.ReLU()(
            1 - lam / norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,), keepdim=True)) *
            self.linear.weight.T.reshape(self.seq_len, -1)).reshape(shape_t).T

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        if ignore_lag:
            GC = norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,))
        else:
            GC = norm(self.linear.weight, dim=(0,))
        if threshold:
            return (GC > 0).int().cpu().numpy()
        else:
            return GC.cpu().numpy()


class CMLPFull(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='sigmoid'):
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

    def regularize(self, lam):
        reg_loss = 0
        for net in self.networks:
            reg_loss += net.regularize(lam)
        return reg_loss

    @torch.no_grad()
    def prox(self, lam):
        for net in self.networks:
            net.prox(lam)

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        return np.stack([net.GC(threshold, ignore_lag) for net in self.networks])


class CMLPwFilter(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1, act='sigmoid'):
        super(CMLPwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size

        # Filters
        self.filter_features = TrainableEltWiseLayer(input_size)
        self.filter_lags = TrainableEltWiseLayer((seq_len, 1))

        # Linear Layer Input
        self.linear = nn.Linear(input_size * seq_len, hidden_dim)

        # LSTM layers
        self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Linear Layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # activation
        self.act = activation_helper(act)

        # linear layer norm
        with torch.no_grad():
            self.norm_const = norm(self.linear.weight).detach()

    def forward(self, input):
        filtered_input = self.filter_lags(self.filter_features(input))
        # normalized weight for linear
        normalized_input = self.norm_const / norm(self.linear.weight) * filtered_input
        flatten_input = normalized_input.reshape(-1, self.seq_len * self.input_size)
        out = self.act(self.linear(flatten_input))
        for layer in self.hidden_s:
            out = self.act(layer(out))
        out = self.fc(out)
        return out

    def regularize(self, lam):
        # Group Lasso on features
        # reg_features = lam * norm(self.filter.weight.reshape(-1, self.input_size), dim=(0,)).sum()
        reg_features = lam * torch.abs(self.filter_features.weight).sum()
        # Group Lasso on time steps
        # reg_lags = lam * norm(self.filter.weight.T.reshape(self.seq_len, -1), dim=(1,)).sum()
        reg_lags = lam * torch.abs(self.filter_lags.weight).sum()
        return reg_features + reg_lags

    @torch.no_grad()
    def prox(self, lam):
        # Group Lasso on features
        # shape = self.filter.weight.shape
        # self.filter.weight.data = (nn.ReLU()(
        #     1 - lam / norm(self.filter.weight.reshape(-1, self.input_size), dim=(0,), keepdim=True)) *
        #                            self.filter.weight.reshape(-1, self.input_size)).reshape(shape)
        self.filter_features.weight.data = nn.ReLU()(
            1 - lam / torch.abs(self.filter_time.weight)) * self.filter_features.weight
        # Group Lasso on time steps
        # shape_t = self.filter.weight.T.shape
        # self.filter.weight.data = (nn.ReLU()(
        #     1 - lam / norm(self.filter.weight.T.reshape(self.seq_len, -1), dim=(1,), keepdim=True)) *
        #                            self.filter.weight.T.reshape(self.seq_len, -1)).reshape(shape_t).T
        self.filter_lags.weight.data = nn.ReLU()(
            1 - lam / torch.abs(self.filter_lags.weight)) * self.filter_time.weight

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        if ignore_lag:
            # GC = norm(self.filter.weight.reshape(-1, self.input_size), dim=(0,))
            GC = torch.abs(self.filter_features.weight)
        else:
            # GC = norm(self.filter.weight, dim=(0,))
            GC = torch.abs(self.filter_lags.weight)
        if threshold:
            return (GC > 0).int().cpu().numpy()
        else:
            return GC.cpu().numpy()


class CMLPwFilterFull(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='sigmoid'):
        super(CMLPwFilterFull, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.networks = nn.ModuleList([
            CMLPwFilter(input_size=input_size, seq_len=seq_len, hidden_dim=hidden_dim,
                        n_layers=n_layers, act=act)
            for _ in range(input_size)])

    def forward(self, input):
        out = torch.cat([network(input) for network in self.networks], dim=1)
        return out

    def regularize(self, lam):
        reg_loss = 0
        for net in self.networks:
            reg_loss += net.regularize(lam)
        return reg_loss

    @torch.no_grad()
    def prox(self, lam):
        for net in self.networks:
            net.prox(lam)

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        return np.stack([net.GC(threshold, ignore_lag) for net in self.networks])


class LeKVAR(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, act='sigmoid', mode=None):
        super(LeKVAR, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.mode = mode

        if mode == "nn":
            # Linear Layer Input
            self.linear = nn.Linear(1, hidden_dim)

            # Middle layers
            self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)])

            # Last Layer
            # Could be extended to different output size (>1)
            self.last = nn.Linear(hidden_dim, 1)

            # linear layer norm
            with torch.no_grad():
                self.norm_const = norm(self.last.weight).detach()

        # Linear Layer
        self.fc = nn.Linear(input_size * seq_len, input_size)

        # activation
        self.act = activation_helper(act)

    def kernel_forward(self, input):
        if self.mode is None:
            return input

        if self.mode == 'nn':
            shape = input.shape
            single_input = input.reshape((*shape, 1))
            out = self.act(self.linear(single_input))
            for layer in self.hidden_s:
                out = self.act(layer(out))
            out = self.norm_const / norm(self.last.weight) * out
            out = self.last(out)
            return out.reshape(shape)

        if self.mode == 'sq':
            return -2 + 1/2 * input**2

    def forward(self, input):
        out = self.kernel_forward(input)
        out = self.fc(out.reshape(-1, self.seq_len * self.input_size))
        return out

    def regularize(self, lam):
        # Group Lasso on features
        reg_features = lam * \
            norm(self.fc.weight.reshape(self.input_size, -1, self.input_size), dim=(1,)).sum()
        # Group Lasso on time steps
        reg_lags = lam * \
            norm(self.fc.weight.T.reshape(self.seq_len, -1, self.input_size), dim=(1,)).sum()
        return reg_features + reg_lags

    @torch.no_grad()
    def prox(self, lam):
        # Group Lasso prox on features
        shape = self.fc.weight.shape
        self.fc.weight.data = (nn.ReLU()(
           1 - lam / norm(self.fc.weight.reshape(self.input_size, -1, self.input_size), dim=(1,), keepdim=True)) *
           self.fc.weight.reshape(self.input_size, -1, self.input_size)).reshape(shape)
        # Group Lasso prox time steps
        shape_t = self.fc.weight.T.shape
        self.fc.weight.data = (nn.ReLU()(
            1 - lam / norm(self.fc.weight.T.reshape(self.seq_len, -1, self.input_size), dim=(1,), keepdim=True)) *
            self.fc.weight.T.reshape(self.seq_len, -1, self.input_size)).reshape(shape_t).T

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        if ignore_lag:
            GC = norm(self.fc.weight.reshape(self.input_size, -1, self.input_size), dim=(1,))
        else:
            GC = torch.abs(self.fc.weight)
            GC = torch.stack([GC[:, self.input_size*i:self.input_size*(i+1)] for i in range(self.seq_len)])
        if threshold:
            return (GC > 0).int().cpu().numpy()
        else:
            return GC.cpu().numpy()


def cmlp(input_size=10, seq_len=10):
    return CMLPFull(input_size, seq_len)


def cmlpwf(input_size=10, seq_len=10):
    return CMLPwFilterFull(input_size, seq_len)


def var(input_size=10, seq_len=10):
    return LeKVAR(input_size, seq_len, mode=None)


def lekvar(input_size=10, seq_len=10):
    return LeKVAR(input_size, seq_len, mode='nn')

def cmlp_s(input_size=10, seq_len=10):
    return CMLPFull(input_size, seq_len, n_layers=1)


def cmlpwf_s(input_size=10, seq_len=10):
    return CMLPwFilterFull(input_size, seq_len, n_layers=1)


def sqvar(input_size=10, seq_len=10):
    return LeKVAR(input_size, seq_len, mode='sq')
