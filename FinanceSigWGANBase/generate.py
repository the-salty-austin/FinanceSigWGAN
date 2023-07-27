"""
Generate data by a trained generator.
"""
import os
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt

GENERATE_FOLDER = 'GeneratedData'


def generate(data, batch_size=1000, device='cpu', foo=lambda x: x):
    experiment_dir = os.path.join('numerical_results', data, 'SigWGAN_LogSigRNN_0')

    # from lib.networks import get_generator
    # generator_config = load_obj(os.path.join(experiment_dir, 'generator_config.pkl'))
    # generator_state_dict = load_obj(os.path.join(experiment_dir, 'generator_state_dict.pt'))
    # generator = get_generator(**generator_config)
    # generator.load_state_dict(generator_state_dict).to(device)

    generator = load_obj(os.path.join(experiment_dir, 'generator_full_model.pt')).to(device)
    data_config = load_obj(os.path.join(experiment_dir, 'data_config.pkl'))
    generator.eval()

    with torch.no_grad():
        x_fake = generator(batch_size, data_config['n_lags'], device)
        x_fake = foo(x_fake)

    x_fake = x_fake.detach().cpu().numpy()

    plt.plot(x_fake[:400, :, 0].T, 'C%s' % 0, alpha=0.1)
    plt.savefig(f'{GENERATE_FOLDER}/{data}_fake_data_{batch_size}.png')
    plt.close()

    x_fake = x_fake[..., 0]
    np.savetxt(f"{GENERATE_FOLDER}/{data}_fake_data_{batch_size}.csv", x_fake,
               delimiter=",", header=",".join(str(i) for i in range(data_config['n_lags'])))
    print(f'Generate {x_fake.shape[0]} sequence data, '
          f'each sequence length {x_fake.shape[1]} with timeframe {data_config["timeframe"]}')


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)


if __name__ == '__main__':
    os.makedirs(GENERATE_FOLDER, exist_ok=True)
    # target_data = 'BINANCE'
    # target_data = 'STABLECOIN'
    # generate(data=target_data, batch_size=100)

    target_dataset = os.listdir('./datasets')
    target_dataset.remove('Uniswap')
    for data in target_dataset:
        generate(data=data, batch_size=2000)
