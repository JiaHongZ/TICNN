import torch.nn as nn
from .functional import elbo
import torch
import pickle


class Elbo(nn.Module):
    
    def __init__(self, alpha_p=1,beta_p=1, kl_weight=1, reduce_samples='mean', annealing='reduce_kl'):
        super(Elbo, self).__init__()

        self.alpha_p = alpha_p #torch.tensor(opt.alpha_p, requires_grad=False)
        self.beta_p = beta_p #torch.tensor(opt.beta_p, requires_grad=False)
        self.w = kl_weight
        self.reduce_samples = reduce_samples

        self.iter = 0.0
        self.base_kl = 0.0 #, requires_grad=False, device=self.alpha_p.device)  #

        # number of batches in epoch (only used for cyclic kl weighting)
        self.M = None

        if annealing == 'no_annealing':
            from .functional import no_annealing as annealing
        elif annealing == 'weight_kl':
            from .functional import weight_kl as annealing
        elif annealing == 'no_kl':
            from .functional import no_kl as annealing
        elif annealing == 'reduce_kl':
            from .functional import reduce_kl as annealing
        elif annealing == 'increase_kl':
            from .functional import increase_kl as annealing
        elif annealing == 'cyclic_kl':
            from .functional import cyclic_kl as annealing
        elif annealing == 'scaled_kl':
            from . functional import scaled_kl as annealing
        else:
            raise NotImplemented

        self.annealing = annealing

    def forward(self, x, beta, label):

        # calculate terms of elbo
        self.nll, kl = elbo(x, beta, label, alpha_p=self.alpha_p, beta_p=self.beta_p, reduction=self.reduce_samples)
        self.kl = kl

        with torch.no_grad():
            if self.iter == 0.0:
                self.base_kl += kl

        # increment counter for each update
        self.iter += 1.0

        # weighting of kl term
        alpha = self.annealing(self.iter, self.M, base_kl=self.base_kl, weight=self.w)

        return self.nll + alpha * self.kl, (self.nll, alpha * self.kl)
