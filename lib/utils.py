import json
import pickle
import random

import numpy as np
import torch


def get_config_path(config, dataset):
    return './configs/{dataset}/{config}.json'.format(config=config, dataset=dataset)


def get_config_path_generator(config, dataset):
    return './configs/{dataset}/generator/{config}.json'.format(
        dataset=dataset, config=config
    )


def get_config_path_discriminator(config, dataset):
    return './configs/{dataset}/discriminator/{config}.json'.format(config=config, dataset=dataset)


def get_sigwgan_experiment_dir(dataset, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, seed=seed)


def get_wgan_experiment_dir(dataset, discriminator, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{discriminator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, discriminator=discriminator, seed=seed)


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


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
