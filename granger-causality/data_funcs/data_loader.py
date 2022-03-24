import os
import torch
import numpy as np

from torch.utils.data import TensorDataset


def load_data(path, dataset, seq_len, device):
    data_path = os.path.join(path, dataset)
    if dataset.startswith('var'):
        train_data = np.load(data_path + '_train.npz')
        x_np = train_data['X']
        trainset = create_dataset_from_ts(x_np, seq_len, device)

        test_data = np.load(data_path + '_test.npz')
        x_np = test_data['X']
        testset = create_dataset_from_ts(x_np, seq_len, device)
    else:
        raise ValueError(f'{dataset} is unknown dataset name.')

    return trainset, testset


def create_dataset_from_ts(x_np, seq_len, device):
    y = torch.tensor(x_np[seq_len:], dtype=torch.float32, device=device)
    x = np.array([x_np[i: i + seq_len] for i in range(len(x_np) - seq_len)])
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return TensorDataset(x, y)
