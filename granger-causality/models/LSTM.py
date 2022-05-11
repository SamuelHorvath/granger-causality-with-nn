import torch
import numpy as np
import torch.nn as nn
from torch.linalg import norm

from .utils import TrainableEltWiseLayer


# TODO: For the current solution, hidden state is not passed beyond sequence length
# for all the LSTM models
# class CLSTM(nn.Module):
#
#     def __init__(self, input_size=10, seq_len=10, hidden_dim=5, n_layers=2, output_size=1, **kwargs):
#         super(CLSTM, self).__init__()
#
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim
#
#         # Linear Layer
#         self.linear = nn.Linear(input_size, hidden_dim)
#
#         # LSTM layers
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers)
#
#         # linear
#         self.fc = nn.Linear(hidden_dim, output_size)
#
#     def forward(self, input, hidden=None):
#         if hidden is None:
#             hidden = self.init_hidden(input.shape[0], input.device)
#         linear_out = self.linear(input)
#         lstm_out, hidden = self.lstm(linear_out, hidden)
#         # Only use the last output of LSTM
#         out = self.fc(lstm_out[:, -1, :].reshape(-1, self.hidden_dim))
#         return out, hidden
#
#     def init_hidden(self, batch_size, device):
#         hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
#                   torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
#         return hidden
#
#     def regularize(self, lam):
#         # Group Lasso on inputs
#         return lam * norm(self.linear.weight, dim=(0, )).sum()
#
#     @torch.no_grad()
#     def prox(self, lam):
#         # Group Lasso on inputs
#         self.linear.weight.data = (nn.ReLU()(
#             1 - lam / norm(self.linear.weight, dim=(0, )), keepdim=True)) * self.linear.weight
#
#     @torch.no_grad()
#     def GC(self, threshold=False, ignore_lag=True):
#         if ignore_lag:
#             GC = norm(self.linear.weight, dim=(0, ))
#         else:
#             return np.array([-1.])
#         if threshold:
#             return (GC > 0).int().cpu().numpy()
#         else:
#             return GC.cpu().numpy()


class CTLSTM(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=5, n_layers=2, output_size=1, lag_pen=False):
        super(CTLSTM, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lag_pen = lag_pen

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, num_layers=self.n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim * seq_len, output_size)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
        lstm_out, hidden = self.lstm(input, hidden)
        out = self.fc(lstm_out.reshape(-1, self.seq_len * self.hidden_dim))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def regularize(self, lam):
        # Group Lasso on inputs
        reg_inputs = lam * norm(self.lstm.weight_ih_l0, dim=(0, )).sum()
        # Hierarchical Group Lasso on outputs
        reg_output = 0.
        if self.lag_pen:
            raise ValueError("Needs to be reimplemented for LSTMs")
            # lam_pen = lam * 30
            # for k in range(self.seq_len):
            #     lower = k * self.hidden_dim
            #     reg_output += lam_pen * norm(self.fc.weight[:, lower:])
        return reg_inputs + reg_output

    @torch.no_grad()
    def prox(self, lam):
        # Group Lasso on inputs
        self.lstm.weight_ih_l0.data = nn.ReLU()(
            1 - lam / norm(self.lstm.weight_ih_l0, dim=(0, ), keepdim=True)) * self.lstm.weight_ih_l0

        # Hierarchical Group Lasso on outputs
        if self.lag_pen:
            raise ValueError("Needs to be reimplemented for LSTMs")
            # for k in range(1, self.seq_len + 1):
            #     lower = k * self.hidden_dim
            #     self.lstm.weight_hh_l0[:, lower:].data = nn.ReLU()(
            #         1 - lam / norm(self.lstm.weight_hh_l0[:, lower:], keepdim=True)) * self.fc.weight[:, lower:]

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        if ignore_lag:
            GC = norm(self.lstm.weight_ih_l0, dim=(0, ))
        else:
            if self.lag_pen:
                raise ValueError("Needs to be reimplemented for LSTMs")
                # GC = torch.stack([norm(self.lstm.weight_hh_l0[:, i * self.hidden_dim: (i+1) * self.hidden_dim])
                #                   for i in range(self.seq_len)])
                # GC = norm(self.fc.weight.T.reshape(self.seq_len, self.hidden_dim), dim=(1,))
            else:
                return np.array([-1.])
        if threshold:
            return (GC > 0).int().cpu().numpy()
        else:
            return GC.cpu().numpy()


class CTLSTMwFilter(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=5, n_layers=2, output_size=1, lag_pen=False):
        super(CTLSTMwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lag_pen = lag_pen

        # Filters
        self.filter_features = TrainableEltWiseLayer(input_size)
        self.filter_lags = TrainableEltWiseLayer((seq_len, 1))

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, num_layers=n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim * seq_len, output_size)

        # linear layer norm
        with torch.no_grad():
            self.norm_const = norm(self.lstm.weight_ih_l0).detach()

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
        filtered_input = self.filter_lags(self.filter_features(input))
        # normalized weight for input weight of lstm
        normalized_input = self.norm_const / norm(self.lstm.weight_ih_l0) * filtered_input
        lstm_out, hidden = self.lstm(normalized_input, hidden)
        out = self.fc(lstm_out.reshape(-1, self.seq_len * self.hidden_dim))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    # TODO: Consider adding option for hierarchical penalty on lags
    def regularize(self, lam):
        # Lasso on inputs
        reg_features = lam * torch.abs(self.filter_features.weight).sum()
        # (Hierarchical) Group Lasso on outputs
        reg_lags = lam * torch.abs(self.filter_lags.weight).sum()
        # if self.lag_pen:
        #     for k in range(1, self.seq_len + 1):
        #         lower = k * self.hidden_dim
        #         reg_output += lam * norm(self.fc.weight[:, lower:])
        return reg_features + reg_lags

    @torch.no_grad()
    def prox(self, lam):
        # Lasso on inputs
        self.filter_features.weight.data = nn.ReLU()(
            1 - lam / torch.abs(self.filter_time.weight)) * self.filter_features.weight

        # (Hierarchical) Group Lasso on outputs
        self.filter_lags.weight.data = nn.ReLU()(
            1 - lam / torch.abs(self.filter_lags.weight)) * self.filter_time.weight
        # if self.lag_pen:
        #     for k in range(self.seq_len):
        #         lower = k * self.hidden_dim
        #         self.fc.weight[:, lower:].data = (nn.ReLU()(
        #             1 - lam / norm(self.fc.weight[:, lower:]), keepdim=True)) * self.fc.weight[:, lower:]

    @torch.no_grad()
    def GC(self, threshold=False, ignore_lag=True):
        if ignore_lag:
            GC = torch.abs(self.filter_features.weight)
        else:
            GC = torch.abs(self.filter_lags.weight)
            # if self.lag_pen:
            #     GC = norm(self.fc.weight.T.reshape(self.seq_len, -1), dim=(1,))
            # else:
            #     return np.array([-1.])
        if threshold:
            return (GC > 0).int().cpu().numpy()
        else:
            return GC.cpu().numpy()


class CMFull(nn.Module):

    def __init__(self, model, input_size=10, seq_len=10, hidden_dim=5, n_layers=2, lag_pen=False):
        super(CMFull, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.networks = nn.ModuleList([
            model(input_size=input_size, seq_len=seq_len,
                  hidden_dim=hidden_dim, n_layers=n_layers,
                  lag_pen=lag_pen)
            for _ in range(input_size)])

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
        out_s = [network(input, hidden[i]) for i, network in enumerate(self.networks)]
        out_x = torch.cat([out[0] for out in out_s], dim=1)
        hidden = tuple([out[1] for out in out_s])
        return out_x, hidden

    def init_hidden(self, batch_size, device):
        hidden = tuple([network.init_hidden(batch_size, device) for network in self.networks])
        return hidden

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


# def clstm(input_size=10, seq_len=10):
#     return CMFull(CLSTM, input_size, seq_len)


def ctlstm(input_size=10, seq_len=10):
    return CMFull(CTLSTM, input_size, seq_len)


def ctlstmwf(input_size=10, seq_len=10):
    return CMFull(CTLSTMwFilter, input_size, seq_len)


def ctlstm_s(input_size=10, seq_len=10):
    return CMFull(CTLSTM, input_size, seq_len, n_layers=1)


def ctlstmwf_s(input_size=10, seq_len=10):
    return CMFull(CTLSTMwFilter, input_size, seq_len, n_layers=1)
