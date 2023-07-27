import json
import os.path
import pickle
import random

import numpy as np
import torch

import glob
from datetime import datetime, timedelta

from pandas import DataFrame
from tqdm import tqdm

from FinanceSigWGANBase.generate import load_obj
from FinanceSigWGANBase.utils.plot import plot_sequence


def get_config_path(config_type, name):
    return os.path.join('configs', config_type, name + '.json')


def get_sigwgan_experiment_dir(dataset, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, seed=seed)


def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """

    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    elif filepath.endswith('json'):
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=4)
        return 0
    else:
        raise NotImplementedError()

    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def rolling_period_resample(dataset, period, n_lags):
    period_dict = {
        'd': 0,
        'h': 0
    }
    assert period[-1] in period_dict.keys()
    period_dict[period[-1]] = int(period[:-1])
    time_period = timedelta(days=period_dict['d'], hours=period_dict['h'])
    time_clip = time_period / n_lags
    dataset_length = len(dataset)

    # print('length', dataset_length)

    def find_data(start_i, target_t):
        i_point = 0
        kk = 0
        while start_i + i_point < dataset_length - 1:
            if dataset[start_i + i_point][0] <= target_t <= dataset[start_i + i_point + 1][0]:
                return start_i + i_point
            else:
                if dataset[start_i + i_point][0] > target_t:
                    print(f'find {target_t} from index {start_i}={dataset[start_i]}. '
                          f'current {dataset[start_i + i_point][0]}')
                    kk += 1
                    if kk > 5:
                        exit()
                i_point += 1

    rolled_dataset = []
    rolled_data_idx = [0] * (n_lags - 1)
    for i, data in enumerate(tqdm(dataset)):
        start_time = datetime.fromtimestamp(int(data[0]))
        end_time = (start_time + time_period).timestamp()
        if dataset[-1][0] < end_time:
            break

        rolled_data = [data]
        time_point = (start_time + time_clip).timestamp()

        data_idx = 0
        for k, idx in enumerate(rolled_data_idx):
            idx = max(idx, data_idx)
            assert dataset[idx][0] < time_point
            data_idx = find_data(idx, time_point)
            rolled_data.append(dataset[data_idx])
            rolled_data_idx[k] = data_idx
            time_point = (start_time + time_clip * len(rolled_data)).timestamp()

        # print(rolled_data_idx)
        rolled_dataset.append(np.stack(rolled_data))

    return np.stack(rolled_dataset)


def rolling_period_resample2(df: DataFrame, period, n_lags):
    period_dict = {
        'd': 24 * 60 * 60,
        'h': 60 * 60
    }
    assert period[-1] in period_dict.keys()
    time_period = int(period[:-1]) * period_dict[period[-1]] / n_lags

    def check_condition(data_frame):
        across_time = data_frame['timestamp'].diff(1).cumsum()
        across_time[0] = 0
        int_time = (across_time / time_period).apply(np.floor)
        idx_ = list(np.where(int_time.diff(1) >= 1)[0])
        output = [0]
        output.extend(idx_)
        return output

    dataset_length = len(df)

    # print('length', dataset_length)
    rolled_dataset = []
    for i in tqdm(range(dataset_length)):

        idx = check_condition(df.iloc[i:])
        if len(idx) < n_lags:
            break
        rolled_dataset.append(df.iloc[idx][df.columns[[-1]]].to_numpy(dtype='float')[:n_lags])

    return np.stack(rolled_dataset)


def plot_individual_data(dataset, locate_dir):
    os.makedirs(locate_dir, exist_ok=True)
    datadir = os.path.join('datasets', dataset)
    files = glob.glob(os.path.join(datadir, '*csv'))
    files = files + glob.glob(os.path.join(datadir, '*', '*csv'))
    # files_name = [f.replace('datasets', '').replace('/', '.') for f in files]
    files_name = [f.split("\\")[-1] for f in files]
    for i, file in enumerate(files):
        dataset = load_obj(file + '.rolled.pt')
        plot_sequence(dataset, os.path.join(locate_dir, files_name[i] + '.png'))
