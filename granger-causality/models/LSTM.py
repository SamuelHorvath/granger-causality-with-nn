import torch
import torch.nn as nn

from .filter import TrainableEltWiseLayer


class CLSTM(nn.Module):

    def __init__(self, input_size=10, hidden_dim=100, n_layers=2, output_size=1):
        super(CLSTM, self).__init__()

        self.n_layers = n_layers

        # Linear Layer
        self.linear = nn.Linear(input_size, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden):
        linear_out = self.linear(input)
        lstm_out, hidden = self.lstm(linear_out, hidden)
        # Only use the last output of LSTM
        out = self.fc(lstm_out[:, -1, :].reshape(-1, self.hidden_dim))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def regularize(self, lam):
        # Group Lasso on inputs
        return lam * torch.linalg.norm(self.linear.weight, dim=(0, )).sum()


class CTLSTM(nn.Module):

    def __init__(self, input_size=10, hidden_dim=100, n_layers=2, seq_len=10, output_size=1):
        super(CTLSTM, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len

        # Linear Layer
        self.linear = nn.Linear(input_size, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim * seq_len, output_size)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
        linear_out = self.linear(input)
        lstm_out, hidden = self.lstm(linear_out, hidden)
        out = self.fc(lstm_out.reshape(-1, self.seq_len * self.hidden_dim))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def regularize(self, lam_1, lam_2):
        # Group Lasso on inputs
        reg_input = lam_1 * torch.linalg.norm(self.linear.weight, dim=(0, )).sum()
        # Hierarchical Group Lasso on outputs
        reg_output = 0.
        for k in range(self.seq_len):
            upper = (k + 1) * self.hidden_dim
            reg_output += lam_2 * torch.linalg.norm(self.fc.weight[:, :upper])
        return reg_input + reg_output


class CTLSTMwFilter(nn.Module):

    def __init__(self, input_size=10, hidden_dim=100, n_layers=2, seq_len=10, output_size=1):
        super(CTLSTMwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len

        # Linear Layer
        self.filter = TrainableEltWiseLayer(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, num_layers=n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim * seq_len, output_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
        filtered_input = self.filter(input)
        lstm_out, hidden = self.lstm(filtered_input, hidden)
        out = self.fc(lstm_out.reshape(-1, self.seq_len * self.hidden_dim))
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def regularize(self, lam_1, lam_2):
        # Group Lasso on inputs
        reg_input = lam_1 * torch.linalg.norm(torch.sigmoid(self.filter.weight), ord=1)
        # Hierarchical Group Lasso on outputs
        reg_output = 0.
        for k in range(self.seq_len):
            upper = (k + 1) * self.hidden_dim
            reg_output += lam_2 * torch.linalg.norm(self.fc.weight[:, :upper])
        return reg_input + reg_output


def clstm_single(input_size=10, hidden_dim=100, n_layers=2):
    return CLSTM(input_size, hidden_dim, n_layers)


def ctlstm_single(input_size=10, hidden_dim=100, n_layers=2, seq_len=10):
    return CTLSTM(input_size, hidden_dim, n_layers, seq_len)


def ctlstmwf_single(input_size=10, hidden_dim=100, n_layers=2, seq_len=10):
    return CTLSTMwFilter(input_size, hidden_dim, n_layers, seq_len)
