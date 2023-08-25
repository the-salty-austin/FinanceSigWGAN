from typing import Tuple, Optional

import signatory
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import math

from lib.augmentations import apply_augmentations, parse_augmentations, Basepoint
from lib.trainers.base import BaseTrainer
from lib.utils import sample_indices
from torch import optim


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class SigWGANTrainer(BaseTrainer):
    # def __init__(self, G, lr, depth, x_real_rolled, augmentations, normalise_sig: bool = True, mask_rate=0.01,
    #              **kwargs):
    def __init__(self, D, G, depth, x_real_rolled: torch.Tensor, augmentations, 
                lr_discriminator, lr_generator, discriminator_steps_per_generator_step,
                normalise_sig: bool = True, mask_rate=0.01, reg_param=10.,
                **kwargs):
        if kwargs.get('augmentations') is not None:
            self.augmentations = kwargs['augmentations']
            del kwargs['augmentations']
        else:
            self.augmentations = None
        
        super(SigWGANTrainer, self).__init__(
            G=G,
            G_optimizer=optim.Adam(G.parameters(), lr=lr_generator),
            **kwargs
        )
        self.sig_w1_metric = SigW1Metric(depth=depth, x_real=x_real_rolled, augmentations=augmentations,
                                         mask_rate=mask_rate, normalise=normalise_sig)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

        self.D_steps_per_G_step = discriminator_steps_per_generator_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(D.parameters(), lr=lr_discriminator, betas=(0, 0.9))  # Using TTUR

        self.reg_param = reg_param
        if self.augmentations is not None:
            self.x_real = apply_augmentations(x_real_rolled, self.augmentations)
        else:
            self.x_real = x_real_rolled

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)
        pbar = tqdm(range(self.n_gradient_steps))
        for _ in pbar:
            self.step(device)
            pbar.set_description(
                "G_loss {:1.6e} D_loss {:1.6e} SigWGAN_GP {:1.6e}".format(self.losses_history['G_loss'][-1],
                                                                       self.losses_history['D_loss'][-1],
                                                                       self.losses_history['SigWGAN_GP'][-1]))

    def step(self, device):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_real_batch = self.x_real[indices].to(device)
            
            # torch.no_grad() is a context-manager that disabled gradient calculation for wrapped code.
            with torch.no_grad():
                x_fake = self.G(
                    batch_size=self.batch_size, n_lags=self.sig_w1_metric.n_lags, device=device
                )

                if self.augmentations is not None:
                    x_fake = apply_augmentations(x_fake, self.augmentations)

            # D_loss_real, D_loss_fake, sigwgan_gp = self.D_trainstep(x_fake, x_real_batch)
            D_loss, sigwgan_gp = self.D_trainstep(x_fake, x_real_batch)

            if i == 0:
                self.losses_history['D_loss'].append(D_loss)
                self.losses_history['SigWGAN_GP'].append(sigwgan_gp)
        
        G_loss = self.G_trainstep(device)
        self.losses_history['G_loss'].append(G_loss)

    def G_trainstep(self, device):
        self.G_optimizer.zero_grad()
        x_fake = self.G(
            batch_size=self.batch_size, n_lags=self.sig_w1_metric.n_lags, device=device
        )

        if self.augmentations is not None:
            x_fake = apply_augmentations(x_fake, self.augmentations)

        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        # d_fake = self.D(x_fake)
        self.D.train()
        # G_loss = self.compute_loss(d_fake, 1)
        G_loss = self.sig_w1_metric( x_fake )
        G_loss.backward()
        self.G_optimizer.step()
        self.evaluate(x_fake)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # x_fake.requires_grad_()
        dloss = self.sig_w1_metric( x_fake )

        # Compute regularizer
        with torch.backends.cudnn.flags(enabled=False):
            wgan_gp = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
        total_loss = dloss + wgan_gp
        total_loss.backward()

        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss.item(), wgan_gp.item()
        # return dloss_real.item(), dloss_fake.item(), wgan_gp.item()

    def compute_loss(self, d_out, target):
        '''
        d_out: real-valued vector / Output from discriminator \n
        target: scalar / 0 or 1
        '''
        # targets = d_out.new_full(size=d_out.size(), fill_value=target)
        targets = d_out.new_full(size = tuple(d_out.size()), fill_value = target)
        res = d_out - targets
        squared_error = sum(res**2) / res.size()[0]
        return squared_error
        # return (2. * targets - 1.) * d_out.mean()
    
    def compute_loss_vector(self, d_out, target):
        '''
        d_out: real-valued vector / Output from discriminator \n
        target: scalar / 0 or 1
        '''
        # targets = d_out.new_full(size=d_out.size(), fill_value=target)
        targets = d_out.new_full(size = tuple(d_out.size()), fill_value = target)
        if target == 1:
            res = targets - d_out
        else:
            res = d_out - targets

        return res

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg

'''
class SigWGANTrainer(BaseTrainer):
    def __init__(self, G, lr, depth, x_real_rolled, augmentations, normalise_sig: bool = True, mask_rate=0.01,
                 **kwargs):
        super(SigWGANTrainer, self).__init__(G=G, G_optimizer=optim.Adam(G.parameters(), lr=lr), **kwargs)
        self.sig_w1_metric = SigW1Metric(depth=depth, x_real=x_real_rolled, augmentations=augmentations,
                                         mask_rate=mask_rate, normalise=normalise_sig)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

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
            # if loss < best_loss:
            #    best_G = deepcopy(self.G.state_dict())
            #    best_loss = loss

        self.G.load_state_dict(self.best_G)  # we retrieve the best generator
'''
        
class SigWGANTrainerDyadicWindows(BaseTrainer):
    def __init__(self, G, lr, depth, x_real_rolled, augmentations, mask_rate=0.01, q=3, **kwargs):
        super(SigWGANTrainerDyadicWindows, self).__init__(G=G, G_optimizer=optim.Adam(G.parameters(), lr=lr), **kwargs)
        self.n_lags = x_real_rolled.shape[1]

        # we create sig-w1-metric for all the dyadic windows
        self.sig_w1_metric = defaultdict(
            list)  # Dictionary that will have key-value pairs (j, sig_w1_metric on the 2^j windows of x_real_rolled)
        # we will only inlcude basepoint when the dyadic window includes the first step of the path
        aug_ = augmentations.copy()
        try:
            aug_.remove(Basepoint())
        except:
            pass

        for j in range(q + 1):
            n_intervals = 2 ** j
            len_interval = x_real_rolled.shape[1] // n_intervals
            for i in range(n_intervals):
                aug = augmentations if i == 0 else aug_
                ind_min = max(0, i * len_interval - 1)
                if i < (n_intervals - 1):
                    self.sig_w1_metric[j].append(
                        SigW1Metric(depth=depth, x_real=x_real_rolled[:, ind_min: (i + 1) * len_interval, :],
                                    augmentations=aug, mask_rate=mask_rate, normalise=True))
                else:
                    self.sig_w1_metric[j].append(
                        SigW1Metric(depth=depth, x_real=x_real_rolled[:, ind_min:, :], augmentations=aug,
                                    mask_rate=mask_rate, normalise=True))

        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

    def fit(self, device):
        self.G.to(device)

        best_loss = 10

        for it in tqdm(range(self.n_gradient_steps)):
            self.G_optimizer.zero_grad()
            x_fake = self.G(
                batch_size=self.batch_size, n_lags=self.n_lags, device=device
            )
            loss = 0

            for j in self.sig_w1_metric.keys():
                # we calculate the sig-w1-metric on each of the 2^j intervals given by the dyadic windows
                len_interval = self.n_lags // 2 ** j
                for i, sig_w1_metric_ in enumerate(self.sig_w1_metric[j]):
                    ind_min = max(0, i * len_interval - 1)
                    if i < len(self.sig_w1_metric[j]) - 1:
                        loss += sig_w1_metric_(x_fake[:, ind_min:(i + 1) * len_interval, :])
                    else:
                        loss += sig_w1_metric_(x_fake[:, ind_min:self.n_lags, :])

            best_loss = loss.item() if it == 0 else best_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5)
            if (it + 1) % 100 == 0: print("sig-w1 loss: {:1.2e}".format(loss.item()))
            self.G_optimizer.step()
            self.scheduler.step()
            self.losses_history['sig_w1_loss'].append(loss.item())
            self.evaluate(x_fake)
            if loss < best_loss:
                best_G = deepcopy(self.G.state_dict())
                best_loss = loss

        self.G.load_state_dict(best_G)  # we retrieve the best generator


def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count:count + dim ** (i + 1)] = expected_signature[
                                                               count:count + dim ** (i + 1)] * math.factorial(i + 1)
            count = count + dim ** (i + 1)
    return expected_signature


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


def masked_rmse(x, y, mask_rate, device):
    mask = torch.FloatTensor(x.shape[0]).to(device).uniform_() > mask_rate
    mask = mask.int()
    return ((x - y).pow(2) * mask).mean().sqrt()


class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, mask_rate: float, augmentations: Optional[Tuple] = (),
                 normalise: bool = True):
        assert len(x_real.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (len(x_real.shape),)

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate
        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(x_real, depth, augmentations, normalise)

    def __call__(self, x_path_nu: torch.Tensor):
        """
        Computes the SigW1 metric.\n
        Equation (4) in 2111.01207
        """
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]
        # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        # y = self.expected_signature_mu.to(device)
        # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        # loss = loss.sum()
        expected_signature_nu = compute_expected_signature(x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
        # loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss
