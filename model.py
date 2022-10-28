import math
import torch
import torch.nn as nn
import numpy as np
import random

dtype = torch.cuda.FloatTensor


def kl_divergence(rho, rho_hat):
    rho_hat = rho_hat.contiguous().view(rho_hat.size(0), -1)
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))


def get_device():
    if torch.cuda.is_available():
        de = 'cuda:0'
    else:
        de = 'cpu'
    return de


device = get_device()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Netlinear1(nn.Module):
    def __init__(self, n_1, n_2, n_3, r=10):
        super(Netlinear1, self).__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.r = r
        self.ueser = nn.Parameter(torch.Tensor(n_1, n_2, int(self.r)))

        self.stdv = 1. / math.sqrt(self.n_1)
        self.film = nn.Linear(int(self.r), int(n_3), bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ueser.data.uniform_(-self.stdv, self.stdv)

    def forward(self):
        out = self.film(self.ueser)
        return out

