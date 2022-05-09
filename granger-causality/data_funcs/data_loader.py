import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, ConcatDataset


def load_data(path, dataset, seq_len):
    data_path = os.path.join(path, dataset)
    if dataset.startswith('var') or dataset.startswith('lorenz'):
        train_data = np.load(data_path + '_train.npz')
        x_np = train_data['X']
        trainset = create_dataset_from_ts(x_np, seq_len)
        GC = train_data['GC']
        if dataset.startswith('var'):
            GC = (GC.sum(axis=0) > 0).astype(int)
        test_data = np.load(data_path + '_test.npz')
        x_np = test_data['X']
        testset = create_dataset_from_ts(x_np, seq_len)
    elif dataset.startswith('InSilicoSize'):
        # time series
        data = pd.read_csv(f'{data_path}-trajectories.tsv', sep='\t')
        time_steps = int(data['Time'].max() / 10. + 1)
        data = data.drop(['Time'], axis=1)
        time_series = int(len(data) / time_steps)
        top_train_index = int(np.ceil(0.8 * time_series))
        trainset = ConcatDataset([create_dataset_from_ts(
            data.loc[i * time_steps: (i + 1) * time_steps].to_numpy(), seq_len)
            for i in np.arange(top_train_index)])
        testset = ConcatDataset([create_dataset_from_ts(
            data.loc[i * time_steps: (i + 1) * time_steps].to_numpy(), seq_len)
            for i in np.arange(top_train_index, time_series)])
        # structure
        dataset_n = dataset.replace('-','_')
        scheme = pd.read_csv(f'{path}DREAM3GoldStandard_{dataset_n}.txt', sep='\t', header=None)
        n = data.shape[1]
        GC = np.ones((n, n), dtype=int)
        for k in range(len(scheme)):
            i = int(scheme.loc[k][0][1:])
            j = int(scheme.loc[k][1][1:])
            val = int(scheme.loc[k][2])
            GC[j - 1, i - 1] = val
    else:
        raise ValueError(f'{dataset} is unknown dataset name.')

    return trainset, testset, GC


def create_dataset_from_ts(x_np, seq_len):
    y = torch.tensor(x_np[seq_len:], dtype=torch.float32)
    x = np.array([x_np[i: i + seq_len] for i in range(len(x_np) - seq_len)])
    x = torch.tensor(x, dtype=torch.float32)
    return TensorDataset(x, y)
