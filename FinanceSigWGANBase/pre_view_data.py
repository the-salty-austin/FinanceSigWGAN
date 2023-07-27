import os
import json
import glob

import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

from FinanceSigWGANBase.utils.datasets import get_dataset


def plot_real_data(dataset, num_data):

    plt.title(dataset)
    with open(f'./configs/{dataset}.json', 'rb') as f:
        data_config = json.load(f)
    dataset = get_dataset(dataset, data_config)
    print('Total data: ', len(dataset))
    dataset = dataset.detach().cpu().numpy()
    idx = np.random.choice(np.arange(len(dataset)), num_data)
    plt.plot(dataset[idx, :, 0].T, 'C%s' % 0, alpha=0.1)
    plt.show()


def get_dataset_time(dataset):
    files = glob.glob(f'./datasets/{dataset}/*csv')
    files = files + glob.glob(f'./datasets/{dataset}/*/*csv')

    for file in files:
        print(file)
        with open(file) as f:
            lines = f.readlines()
            print('length: ', len(lines) - 1)
            i = 0
            last_line = lines[1]
            last_dt = datetime.now()
            for line in lines[1:]:
                if line == last_line:
                    continue
                elif i > 10:
                    break
                else:
                    last_line = line
                    i += 1
                timestamp = int(line.split(',')[0])
                dt_obj = timestamp2time(timestamp)
                print(f"date_time: {dt_obj} delta {dt_obj - last_dt}")
                last_dt = dt_obj

            timestamp = int(lines[-1].split(',')[0])
            dt_obj = timestamp2time(timestamp)
            print(f"date_time: {dt_obj} delta {dt_obj - datetime.now()}")
        print('='*30)


def timestamp2time(timestamp):
    try:
        dt_obj = datetime.fromtimestamp(timestamp)
    except OSError:
        dt_obj = datetime.fromtimestamp(timestamp / 1000)
    return dt_obj


if __name__ == '__main__':

    dirs = os.listdir('datasets')
    data = 'STABLECOIN'
    # data = 'BINANCE'
    # data = 'TetherUSD'
    # data = 'Uniswap'

    # get_dataset_time(data)

    # plot_real_data(data, 600)

    # dirs = ['Uniswap', ]
    #
    for data in dirs:
        print(data)
        # get_dataset_time(data)

        plot_real_data(data, 500)
