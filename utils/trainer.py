
from tqdm import tqdm
from lib.trainers.base import BaseTrainer
from lib.trainers.sig_wgan import SigW1Metric

from torch import optim
import numpy as np


class SigWGANTrainer(BaseTrainer):
    def __init__(self, generator, lr, depth, x_real_rolled, augmentations, normalise_sig: bool = True, mask_rate=0.01,
                 es_patience=500, es_min_delta=0.05, es_target='sig_w1_loss', **kwargs):
        super(SigWGANTrainer, self).__init__(G=generator,
                                             G_optimizer=optim.Adam(generator.parameters(), lr=lr), **kwargs)
        self.sig_w1_metric = SigW1Metric(depth=depth, x_real=x_real_rolled, augmentations=augmentations,
                                         mask_rate=mask_rate, normalise=normalise_sig)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)
        self.early_stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)
        self.es_target = es_target

    def fit(self, device):
        self.G.to(device)
        best_loss = None
        pbar = tqdm(range(self.n_gradient_steps))
        for j in pbar:
            self.G_optimizer.zero_grad()
            x_fake = self.G(
                batch_size=self.batch_size, n_lags=self.sig_w1_metric.n_lags, device=device
            )
            loss = self.sig_w1_metric(x_fake)
            loss.backward()
            best_loss = loss.item() if j == 0 else best_loss

            pbar.set_description("sig-w1 loss: {:1.6e}".format(loss.item()))
            self.G_optimizer.step()
            self.scheduler.step()
            self.losses_history['sig_w1_loss'].append(loss.item())
            self.evaluate(x_fake)

            if self.early_stopper.early_stop(self.losses_history[self.es_target][-1]):
                break

            # if loss < best_loss:
            #    best_G = deepcopy(self.G.state_dict())
            #    best_loss = loss

        self.G.load_state_dict(self.best_G)  # we retrieve the best generator


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
