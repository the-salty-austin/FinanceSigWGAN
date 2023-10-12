import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
from torch.utils.data import RandomSampler

from generate import load_obj
from lib.utils import sample_indices
from utils.utils import rolling_period_resample, save_obj

from lib.datasets import get_rBergomi_paths

DATA_DIR = 'datasets'


def rolling_window(x: torch.Tensor, n_lags: int):
    """
    ex
    """
    # https://www.geeksforgeeks.org/python-pytorch-stack-method/
    return torch.stack([x[t:t + n_lags, :] for t in range(x.shape[0] - n_lags + 1)], dim=0)


def transfer_percentage_seq(x):
    start = x[:, 0 :1, :]
    # remove zero start
    idx_ = torch.nonzero(start == 0, as_tuple=False).tolist()
    if idx_:
        idx_ = idx_[0]
    idx_ = list(set(list(range(x.shape[0]))) - set(idx_))

    new_x = x[idx_, ...]
    new_start = start[idx_, ...]

    new_x = (new_x - new_start) / new_start
    return new_x


def get_smallest_data_size(data):
    min_size = np.inf
    for d in data:
        min_size = min(d.shape[0], min_size)
    
    return min_size

def get_data_stablecoin(datadir, data_config):
    """
    Get Stable Coin dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, n_lags, 1|2)
    """
    files = glob.glob(os.path.join(datadir, '*csv'))
    data_path = os.path.join(datadir, 'x_real_rolled.pt')
    if os.path.exists(data_path):
        print('Use data: \n\t' + '\n\t'.join(files))
        datasets = load_obj(data_path)
        print(f'\tRolled data for training, shape {list(datasets.shape)}')
        # print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in datasets[0][:3]]))
        print('\tExample :', datasets[0][:3])
    else:
        datasets = []
        for file in files:
            if os.path.exists(file + '.rolled.pt'):
                dataset = load_obj(data_path)
            else:
                decimals = int(file.split('_')[-2])
                df = pd.read_csv(file)
                dataset = df.drop_duplicates(subset=[df.columns[0]], keep='last')

                dataset = dataset.to_numpy()
                # print(dataset)
                print(f'Preprocess data: {os.path.basename(file)}, shape {df.shape}, after del repeat {dataset.shape}')
                dataset = dataset[:, 1:2].astype(np.float) / 10 ** decimals
                dataset = rolling_window(torch.FloatTensor(dataset), data_config['n_lags'])
                # print(dataset)
                dataset = transfer_percentage_seq(dataset)
                # print(dataset)

                save_obj(dataset, file + '.rolled.pt')
            datasets.append(dataset)
            print(f'\tRolled data for training, shape {list(dataset.shape)}')
            print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
            print()
        # datasets = torch.cat(datasets, dim=0)
        smallest = get_smallest_data_size(datasets)
        datasets = [x[:smallest,:] for x in datasets]
        datasets = torch.cat(datasets, dim=2)
        save_obj(datasets, data_path)
    return datasets


def get_data_binance(datadir, data_config):
    """
    Get ETH vs USTD dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, n_lags, 1|2)
    """
    files = glob.glob(os.path.join(datadir, '*csv'))
    data_path = os.path.join(datadir, 'x_real_rolled.pt')
    if os.path.exists(data_path):
        print('Use data: \n\t' + '\n\t'.join(files))
        datasets = load_obj(data_path)
        print(f'\tRolled data for training, shape {list(datasets.shape)}')
        print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in datasets[0][:3]]))
    else:
        datasets = []
        for file in files:
            if os.path.exists(file + '.rolled.pt'):
                dataset = load_obj(data_path)
            else:
                df = pd.read_csv(file)
                print(f'Preprocess data: {os.path.basename(file)}, shape {df.shape}')
                dataset = df[df.columns[data_config['use_columns']]].to_numpy(dtype='float')
                dataset = torch.FloatTensor(dataset)
                dataset = rolling_window(dataset, data_config['n_lags'])
                dataset = transfer_percentage_seq(dataset)

                save_obj(dataset, file + '.rolled.pt')
            datasets.append(dataset)
            print(f'\tRolled data for training, shape {list(dataset.shape)}')
            print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
            print()
        datasets = torch.cat(datasets, dim=0)
        save_obj(datasets, data_path)
    return datasets


def get_data_mybinance(datadir, data_config):
    """
    Get ETH vs USTD dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, n_lags, 1|2)
    """
    files = glob.glob(os.path.join(datadir, '*csv'))
    data_path = os.path.join(datadir, 'x_real_rolled.pt')
    if os.path.exists(data_path):
        print('Use data: \n\t' + '\n\t'.join(files))
        datasets = load_obj(data_path)
        print(f'\tRolled data for training, shape {list(datasets.shape)}')
        # print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in datasets[0][:3]]))
        print('\tExample :', datasets[0][:3])
    else:
        datasets = []
        for file in files:
            if os.path.exists(file + '.rolled.pt'):
                dataset = load_obj(data_path)
            else:
                df = pd.read_csv(file)
                print(df)
                print(f'Preprocess data: {os.path.basename(file)}, shape {df.shape}')
                dataset = df[df.columns[data_config['use_columns']]].to_numpy(dtype='float')
                print(dataset)
                print(df.columns[data_config['use_columns']])
                dataset = torch.FloatTensor(dataset)
                dataset = rolling_window(dataset, data_config['n_lags'])
                dataset = transfer_percentage_seq(dataset)

                save_obj(dataset, file + '.rolled.pt')
            datasets.append(dataset)
            print(f'\tRolled data for training, shape {list(dataset.shape)}')
            print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
            print()
        datasets = torch.cat(datasets, dim=2)
        save_obj(datasets, data_path)
    return datasets


def get_data_coin_gecko(datadir, data_config):
    """
    Get ETH vs USTD dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, n_lags, 1|2)
    """
    files = glob.glob(os.path.join(datadir, '*csv'))
    data_path = os.path.join(datadir, 'x_real_rolled.pt')
    if os.path.exists(data_path):
        print('Use data: \n\t' + '\n\t'.join(files))
        datasets = load_obj(data_path)
        print(f'\tRolled data for training, shape {list(datasets.shape)}')
        print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in datasets[0][:3]]))
    else:
        datasets = []
        for file in files:
            if os.path.exists(file + '.rolled.pt'):
                dataset = load_obj(data_path)
            else:
                df = pd.read_csv(file)
                print(f'Preprocess data: {os.path.basename(file)}, shape {df.shape}')
                dataset = df.to_numpy(dtype='float')[:, 1:2]
                dataset = torch.FloatTensor(dataset)
                dataset = rolling_window(dataset, data_config['n_lags'])
                dataset = transfer_percentage_seq(dataset)

                save_obj(dataset, file + '.rolled.pt')
            datasets.append(dataset)
            print(f'\tRolled data for training, shape {list(dataset.shape)}')
            print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
            print()
        datasets = torch.cat(datasets, dim=0)
        save_obj(datasets, data_path)
    return datasets


def get_data_uni_swap(datadir, data_config):
    """
    Get ETH vs USTD dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, n_lags, 1|2)
    """

    sub_folders = os.listdir(datadir)
    datasets = []
    for sub_folder in sub_folders:
        files = glob.glob(os.path.join(datadir, sub_folder, '*csv'))
        data_path = os.path.join(datadir, sub_folder, 'x_real_rolled.pt')
        if os.path.exists(data_path):
            print('Use data: \n\t' + '\n\t'.join(files))
            dataset = load_obj(data_path)
            datasets.append(dataset)
            print(f'\tRolled data for training, shape {list(dataset.shape)}')
            print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
        else:
            sub_datasets = []
            for file in files:
                if os.path.exists(file + '.rolled.pt'):
                    dataset = load_obj(file + '.rolled.pt')
                else:
                    df = pd.read_csv(file)
                    print(f'Preprocess data: {os.path.basename(file)}, shape {df.shape}')
                    df = df.sort_values(by=df.columns[0])
                    df = df.drop_duplicates(subset=[df.columns[0]], keep='last')
                    df = df.dropna(subset=[df.columns[-1]])
                    dataset = df[df.columns[[0, -1]]].to_numpy(dtype='float')
                    dataset = rolling_period_resample(dataset, data_config['period'], data_config['n_lags'])

                    dataset = torch.FloatTensor(dataset[..., 1:2])
                    dataset = transfer_percentage_seq(dataset)
                    save_obj(dataset, file + '.rolled.pt')
                sub_datasets.append(dataset)
                print(f'\tRolled data for training, shape {list(dataset.shape)}')
                print('\tExample : {:.6f}, {:.6f}, {:.6f} ...'.format(*[i.item() for i in dataset[0][:3]]))
                print()
            sub_datasets = torch.cat(sub_datasets, dim=0)
            save_obj(sub_datasets, data_path)
            datasets.append(sub_datasets)
    datasets = torch.cat(datasets, dim=0)
    print('Original data amount: ', len(datasets))
    datasets_ids = list(RandomSampler(datasets, replacement=True, num_samples=4000000))
    datasets = datasets[datasets_ids, ...]
    return datasets


def get_corr_brownian_motion(datadir, data_config):

    data_path = os.path.join(datadir, 'x_real_rolled.pt')

    n_lag = data_config['n_lags']
    d = data_config['n_path_dim']

    cov_matrix = np.random.rand(d, d)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T) # Ensure it's symmetric
    np.fill_diagonal(cov_matrix, 1.0) # Make sure the diagonal elements are all ones
    std_deviation = np.sqrt(np.diag(cov_matrix))
    C = cov_matrix / np.outer(std_deviation, std_deviation)

    L = np.linalg.cholesky(C) # Cholesky Decomposition
    dt = 1/252 # daily

    tensor_list = []

    for _ in range(data_config['n_paths']):
        X = np.random.normal(0, dt**(1/2), (d, n_lag)) # Brownian Motion - n_lag daily time steps for 2 assets
        CX = np.dot(L, X) # correlated paths

        tensor = torch.from_numpy( CX.cumsum(1) ).T
        tensor_list.append( tensor )

    datasets = torch.stack(tensor_list, dim=0)

    save_obj(datasets, data_path)
    return datasets


def get_dataset(dataset: str, data_config: dict):
    """
    Loads different datasets and downloads missing datasets.

    Parameters
    ----------
    dataset: str, specifies which dataset loading function to use
    data_config: dict, contains kwargs for loading the dataset
    Returns
    -------

    x_real: torch.Tensor, dataset

    """
    if dataset == 'STABLECOIN':
        x_real = get_data_stablecoin(os.path.join(DATA_DIR, 'STABLECOIN'), data_config=data_config)
    elif dataset == 'MyBinance':
        x_real = get_data_mybinance(os.path.join(DATA_DIR, 'MyBinance'), data_config=data_config)
    elif dataset == 'BINANCE':
        x_real = get_data_binance(os.path.join(DATA_DIR, 'BINANCE'), data_config=data_config)
    elif dataset == 'CoinGecko':
        x_real = get_data_coin_gecko(os.path.join(DATA_DIR, 'CoinGecko'), data_config=data_config)
    elif dataset == 'TetherUSD':
        x_real = get_data_coin_gecko(os.path.join(DATA_DIR, 'TetherUSD'), data_config=data_config)
    elif dataset == 'WrappedBitcoin':
        x_real = get_data_coin_gecko(os.path.join(DATA_DIR, 'WrappedBitcoin'), data_config=data_config)
    elif dataset == 'StakedETH':
        x_real = get_data_uni_swap(os.path.join(DATA_DIR, 'StakedETH'), data_config=data_config)
    elif dataset == 'Uniswap':
        x_real = get_data_uni_swap(os.path.join(DATA_DIR, 'Uniswap'), data_config=data_config)
    elif dataset == 'CorrelatedBrownian':
        x_real = get_corr_brownian_motion(os.path.join(DATA_DIR, 'CorrelatedBrownian'), data_config=data_config)
    elif dataset == 'RoughVolatility':
        x_real = get_rBergomi_paths()
    
    else:
        raise NotImplementedError('Dataset %s not valid' % dataset)

    assert len(x_real.shape) == 3

    return x_real.float()


def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float
):
    """
    Apply a train-test split to a given tensor along the first dimension of the tensor.

    Parameters
    ----------
    x: torch.Tensor, tensor to split.
    train_test_ratio: percentage of samples kept in train set, i.e. 0.8 => keep 80% of samples in the train set

    Returns
    -------
    x_train: torch.Tensor, training set
    x_test: torch.Tensor, test set
    """
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test
