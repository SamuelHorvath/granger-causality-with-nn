import torch
import torch.nn as nn

from .utils import TrainableEltWiseLayer


# TODO: For current solution, hidden state is not passed beyond sequence length
# for all the LSTM models
class CLSTM(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1):
        super(CLSTM, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Linear Layer
        self.linear = nn.Linear(input_size, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers)

        # linear
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape[0], input.device)
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

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1):
        super(CTLSTM, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

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

    def __init__(self, input_size=10, seq_len=10, hidden_dim=10, n_layers=2, output_size=1):
        super(CTLSTMwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

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


class CMFull(nn.Module):

    def __init__(self, model, input_size=10, seq_len=10, hidden_dim=10, n_layers=2):
        super(CMFull, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.networks = nn.ModuleList([
            model(input_size=input_size, seq_len=seq_len,
                  hidden_dim=hidden_dim, n_layers=n_layers)
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


def clstm(input_size=10, seq_len=10):
    return CMFull(CLSTM, input_size, seq_len)


def ctlstm(input_size=10, seq_len=10):
    return CMFull(CTLSTM, input_size, seq_len)


def ctlstmwf(input_size=10, seq_len=10):
    return CMFull(CTLSTMwFilter, input_size, seq_len)
