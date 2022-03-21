import torch
import torch.nn as nn

from .filter import TrainableEltWiseLayer


class CMLP(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=100, n_layers=2, output_size=1):
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
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.relu(self.linear(input))
        for layer in self.hidden_s:
            out = self.relu(layer(out))
        out = self.fc(out)
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time


class CMLPwFilter(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=100, n_layers=2, output_size=1):
        super(CMLPwFilter, self).__init__()

        self.n_layers = n_layers
        self.seq_len = seq_len
        self.input_size = input_size

        # Filter
        self.filter = TrainableEltWiseLayer(input_size)
        # Linear Layer Input
        self.linear = nn.Linear(input_size, hidden_dim)

        # LSTM layers
        self.hidden_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Linear Layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # activation
        self.relu = nn.ReLU()

    def forward(self, input):
        filtered_input = self.filter(input)
        out = self.relu(self.linear(filtered_input))
        for layer in self.hidden_s:
            out = self.relu(layer(out))
        out = self.fc(out)
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.linear.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.linear.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time


class LeKVAR(nn.Module):

    def __init__(self, input_size=10, seq_len=10, hidden_dim=100, n_layers=2, output_size=1):
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
        self.fc = nn.Linear(hidden_dim * seq_len, output_size)

        # activation
        self.relu = nn.ReLU()

    def forward(self, input):
        input_times = input.reshape(-1, self.seq_len, self.input_size)
        out = self.linear(input_times)
        for layer in self.hidden_s:
            out = self.relu(out)
            out = layer(out)
        out = self.fc(out.reshape(-1, self.seq_len * self.hidden_dim))
        return out

    def regularize(self, lam_1, lam_2):
        # Group Lasso on features
        reg_features = lam_1 * torch.linalg.norm(self.fc.weight.reshape(-1, self.input_size), dim=(0,))
        # Group Lasso on time steps
        reg_time = lam_2 * torch.linalg.norm(self.fc.weight.T.reshape(self.seq_len, -1), dim=(1,))
        return reg_features + reg_time
