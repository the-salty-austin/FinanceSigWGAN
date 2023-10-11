"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""

import os
import torch
import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt
from typing import Optional

from evaluate import evaluate_generator
from lib.networks import get_generator, get_discriminator
from lib.utils import to_numpy, load_obj
from lib.augmentations import parse_augmentations
from lib.test_metrics import get_standard_test_metrics
from lib.trainers.sig_wgan import compute_expected_signature

from lib.trainers.wgan import WGANTrainer
from lib.trainers.sig_wgan import SigWGANTrainer
from utils.plot import plot_signature, plot_test_metrics
from utils.datasets import get_dataset, train_test_split
from utils.utils import set_seed, save_obj, get_experiment_dir, get_sigwgan_experiment_dir, get_config_path, \
    plot_individual_data

os.environ['PYTHONHASHSEED'] = "0"


def main(
        data_config: dict,
        dataset: str,
        experiment_dir: str,
        gan_algo: str,
        gan_config: dict,
        generator_config: dict,
        discriminator_config: Optional[dict] = None,
        device: str = 'cpu',
        seed: Optional[int] = 0
):
    """

    Full training procedure.
    Includes: initialising the dataset / generator / GAN and training the GAN.
    """

    n_lags = data_config["n_lags"]

    # Get / prepare dataset
    x_real_rolled = get_dataset(dataset, data_config)
    x_real_rolled = x_real_rolled.to(device)
    set_seed(seed)
    print('Total data: ', list(x_real_rolled.shape))
    # exit()
    x_real_train, x_real_test = train_test_split(x_real_rolled, train_test_ratio=0.8)
    x_real_dim: int = x_real_rolled.shape[2]

    # Compute test metrics for train and test set
    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test)

    # Get generator
    set_seed(seed)
    generator_config.update(output_dim=x_real_dim)
    G = get_generator(**generator_config).to(device)

    print( "generator_config:", generator_config )

    # Get GAN
    if gan_algo == 'SigWGAN':
        trainer = SigWGANTrainer(G,
                                 x_real_rolled=x_real_rolled,
                                 test_metrics_train=test_metrics_train,
                                 test_metrics_test=test_metrics_test,
                                 **gan_config
                                )
    elif gan_algo == 'WGAN':
        discriminator_config.update(input_dim=x_real_dim * n_lags)
        D = get_discriminator(**discriminator_config)
        trainer = WGANTrainer(D, G,
                                 x_real=x_real_rolled,
                                 test_metrics_train=test_metrics_train,
                                 test_metrics_test=test_metrics_test,
                                 **gan_config
                            )
    else:
        raise NotImplementedError()

    # Start training
    set_seed(seed)
    trainer.fit(device=device)

    # Store relevant training results
    # save_obj(to_numpy(x_real_rolled), os.path.join(experiment_dir, 'x_real_rolled.pkl'))
    # save_obj(to_numpy(x_real_train), os.path.join(experiment_dir, 'x_real_train.pkl'))
    save_obj(to_numpy(x_real_test), os.path.join(experiment_dir, 'x_real_test.pkl'))
    save_obj(trainer.losses_history, os.path.join(experiment_dir, 'losses_history.pkl'))  # dev of losses / metrics
    save_obj(trainer.G.state_dict(), os.path.join(experiment_dir, 'generator_state_dict.pt'))
    save_obj(trainer.G, os.path.join(experiment_dir, 'generator_full_model.pt'))
    save_obj(generator_config, os.path.join(experiment_dir, 'generator_config.pkl'))

    loss_history = os.path.join(experiment_dir, 'LossHistory')
    os.makedirs(loss_history, exist_ok=True)

    if gan_algo == 'SigWGAN':
        plt.plot(trainer.losses_history['sig_w1_loss'], alpha=0.8)
        plt.grid()
        plt.yscale('log')
        plt.savefig(os.path.join(loss_history, 'sig_loss.png'))
        plt.close()
    elif gan_algo == 'WGAN':
        # plt.plot(trainer.losses_history['D_loss_fake'], label="D_loss_fake")
        # plt.plot(trainer.losses_history['D_loss_real'], label="D_loss_real")
        plt.plot(trainer.losses_history['G_loss'], label="G_loss")
        plt.plot(np.array(trainer.losses_history['D_loss_fake']) - np.array(trainer.losses_history['D_loss_real']) + np.array(trainer.losses_history['WGAN_GP']), label="D_loss" )
        plt.grid()
        # plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig(os.path.join(loss_history, 'wgan_loss.png'))
        plt.close()

    plot_test_metrics(trainer.test_metrics_train, trainer.losses_history, 'train', locate_dir=loss_history)

    plot_test_metrics(trainer.test_metrics_train, trainer.losses_history, 'test', locate_dir=loss_history)

    with torch.no_grad():
        x_fake = G(1024, n_lags, device)

    for i in range(x_real_dim):
        plt.plot(to_numpy(x_fake[:400, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(os.path.join(experiment_dir, 'x_fake.png'))
    plt.close()

    for i in range(x_real_dim):
        random_indices = torch.randint(0, x_real_rolled.shape[0], (400,))
        plt.plot(to_numpy(x_real_rolled[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(os.path.join(experiment_dir, 'x_real.png'))
    plt.close()

    plot_individual_data(dataset=data_config['name'], locate_dir=os.path.join(experiment_dir, 'RealDataFigure'))

    evaluate_generator(experiment_dir, batch_size=5000, )

    if gan_algo == 'SigWGAN':
        plot_signature(trainer.sig_w1_metric.expected_signature_mu)
        plt.savefig(os.path.join(experiment_dir, 'sig_real.png'))
        plt.close()

        plot_signature(trainer.sig_w1_metric.expected_signature_mu)
        plot_signature(compute_expected_signature(x_fake,
                                                trainer.sig_w1_metric.depth, trainer.sig_w1_metric.augmentations))
        plt.savefig(os.path.join(experiment_dir, 'sig_real_fake.png'))
        plt.close()


def benchmark_sigwgan(
        datasets=('BINANCE', 'STABLECOIN'),
        generators=('LogSigRNN', 'LSTM'),
        n_seeds={"start": 0,"end": 2,"step": 1},
        device='cuda:0',
):
    """ Benchmark for SigWGAN. """
    seeds = list(range(n_seeds["start"],n_seeds["end"],n_seeds["step"]))

    grid = itertools.product(datasets, generators, seeds)

    for dataset, generator, seed in grid:
        print(f"SigWGAN - data:{dataset}, G:{generator}, seed:{seed}")
        data_config = load_obj(get_config_path('', dataset))
        gan_config = load_obj(get_config_path('Trainer', 'trainer_SigWGAN'))
        generator_config = load_obj(get_config_path('Generator', 'gen_' + generator))

        experiment_dir = get_sigwgan_experiment_dir(dataset, generator, 'SigWGAN', seed)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        save_obj(data_config, os.path.join(experiment_dir, 'data_config.json'))
        save_obj(gan_config, os.path.join(experiment_dir, 'gen_config.json'))
        save_obj(generator_config    , os.path.join(experiment_dir, 'generator_config.json'))
        
        if gan_config.get('augmentations') is not None:
            gan_config['augmentations'] = parse_augmentations(gan_config.get('augmentations'))

        if generator_config.get('augmentations') is not None:
            generator_config['augmentations'] = parse_augmentations(generator_config.get('augmentations'))

        if generator_config['generator_type'] == 'LogSigRNN':
            generator_config['n_lags'] = data_config['n_lags']

        save_obj(data_config, os.path.join(experiment_dir, 'data_config.pkl'))
        save_obj(gan_config, os.path.join(experiment_dir, 'gen_config.pkl'))
        save_obj(generator_config, os.path.join(experiment_dir, 'generator_config.pkl'))

        print('Training: %s' % experiment_dir.split('/')[-2:])

        main(
            dataset=dataset,
            data_config=data_config,
            device=device,
            experiment_dir=experiment_dir,
            gan_algo='SigWGAN',
            seed=seed,
            gan_config=gan_config,
            generator_config=generator_config,
        )


def benchmark_wgan(
    datasets=('BINANCE', 'STABLECOIN'),
    generators=('NSDE', 'LSTM'),
    discriminators=('ResFNN',),
    n_seeds={"start": 0,"end": 2,"step": 1},
    device='cuda:0',
):
    """ Benchmark for WGAN. """
    seeds = list(range(n_seeds["start"],n_seeds["end"],n_seeds["step"]))

    grid = itertools.product(datasets, discriminators, generators, seeds)

    for dataset, discriminator, generator, seed in grid:
        print(f"data:{dataset}, G:{generator}, D:{discriminator}, seed:{seed}")

        data_config = load_obj(get_config_path('', dataset))
        gan_config = load_obj(get_config_path('Trainer', 'trainer_WGAN'))
        generator_config = load_obj(get_config_path('Generator', 'gen_' + generator))
        discriminator_config = load_obj(get_config_path('Discriminator', discriminator))

        experiment_dir = get_experiment_dir(dataset, generator, discriminator, 'WGAN', seed)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        if gan_config.get('augmentations') is not None:
            gan_config['augmentations'] = parse_augmentations(gan_config.get('augmentations'))

        if generator_config.get('augmentations') is not None:
            generator_config['augmentations'] = parse_augmentations(generator_config.get('augmentations'))

        if generator_config['generator_type'] == 'LogSigRNN':
            generator_config['n_lags'] = data_config['n_lags']        

        save_obj(data_config, os.path.join(experiment_dir, 'data_config.pkl'))
        save_obj(gan_config, os.path.join(experiment_dir, 'gan_config.pkl'))
        save_obj(generator_config, os.path.join(experiment_dir, 'generator_config.pkl'))
        save_obj(discriminator_config, os.path.join(experiment_dir, 'discriminator_config.pkl'))

        print('Training: %s' % experiment_dir.split('/')[-2:])

        main(
            dataset=dataset,
            data_config=data_config,
            device=device,
            experiment_dir=experiment_dir,
            gan_algo='WGAN',
            seed=seed,
            gan_config=gan_config,
            generator_config=generator_config,
            discriminator_config=discriminator_config,
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    if torch.cuda.is_available():
        compute_device = 'cuda:{}'.format(args.device)
    else:
        compute_device = 'cpu'
    # target_dataset = os.listdir('./datasets')[-1:]
    target_dataset = ('MyBinance', )
    # target_dataset.remove('Uniswap')
    # target_dataset.append('Uniswap')
    # target_dataset = ('BINANCE',)
    # target_dataset = ('WrappedBitcoin',)
    # target_dataset = ('STABLECOIN',)
    # target_dataset = ('Uniswap',)
    # target_dataset = ('BINANCE', 'STABLECOIN')
    training_generators = ('LogSigRNN',)
    training_discriminators=('ResFNN',)
    # training_generators = ('LSTM',)
    # training_generators = ('LogSigRNN', 'LSTM')
  
   
     
    n_seeds = {
        "start": 0,
        "end": 1,
        "step": 5
    }

    benchmark_sigwgan(datasets=target_dataset,
                      generators=training_generators,
                      n_seeds=n_seeds,
                      device=compute_device)
    
    # benchmark_wgan(datasets=target_dataset,
    #                generators=training_generators,
    #                discriminators=training_discriminators,
    #                n_seeds=n_seeds,
    #                device=compute_device)
