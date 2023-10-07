import os
from matplotlib import pyplot as plt

from lib.test_metrics import *
from lib.utils import to_numpy


def plot_signature(sig):
    plt.plot(to_numpy(sig).T, 'o')

# def plot_signature(signature_tensor, alpha=0.2):
#     plt.plot(to_numpy(signature_tensor).T, alpha=alpha, linestyle='None', marker='o')
#     plt.grid()


def plot_test_metrics(metrics, losses_history, mode, locate_dir):
    os.makedirs(locate_dir, exist_ok=True)

    for i, test_metric in enumerate(metrics):
        name = test_metric.name
        loss = losses_history[name + '_' + mode]
        try:
            loss = np.concatenate(loss, 1).T
        except ValueError:
            loss = np.array(loss)

        plt.plot(loss, label=name)
        plt.grid()
        plt.legend()
        plt.ylim(bottom=0.)
        plt.xlabel('Number of generator weight updates')
        plt.savefig(os.path.join(locate_dir, 'loss_development_{}_{}.png').format(mode, name))
        plt.close()


def plot_sequence(x_sequence, path, sample_num=200):
    x_real_dim = x_sequence.shape[-1]
    for i in range(x_real_dim):
        random_indices = torch.randint(0, x_sequence.shape[0], (sample_num,))
        plt.plot(to_numpy(x_sequence[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(path)
    plt.close()
