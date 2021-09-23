import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssl(y_pred, y_real, window, window_size, channel, beta=0.1, gamma=0.5, eps=1e-6):
    y = y_pred.sigmoid() + eps
    cross_entropy = y_pred - y_real * y_pred + (1 + torch.exp(-y_pred)).log()

    mu1 = F.conv2d(y, window, padding = window_size//2, groups = 1)
    mu2 = F.conv2d(y_real, window, padding = window_size//2, groups = 1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(y*y, window, padding = window_size//2, groups = 1) - mu1_sq
    sigma2_sq = F.conv2d(y_real*y_real, window, padding = window_size//2, groups = 1) - mu2_sq

    C1 = 0.01

    e = abs((y_real - mu1 + C1)/(torch.sqrt(sigma1_sq + C1) + C1) - (y - mu2 + C1)/(torch.sqrt(sigma2_sq + C1) + C1))
    e_max = torch.max(e)
  
    mask = torch.where(e > beta * e_max, 1, 0)
    M = torch.sum(mask)
    mask = torch.where(e > beta * e_max, 1, 0) * e

    ssl = (mask * cross_entropy).sum() / M

    return gamma * cross_entropy.mean() + (1 - gamma) * ssl

class SSL(torch.nn.Module):
    def __init__(self, window_size = 11, beta=0.1, gamma=0.5, eps=1e-6):
        super(SSL, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.beta=beta
        self.gamma=gamma
        self.eps=eps

    def forward(self, y_pred, y_real):
        (_, channel, _, _) = y_real.size()

        if channel == self.channel and self.window.data.type() == y_real.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if y_real.is_cuda:
                window = window.cuda(y_real.get_device())
            window = window.type_as(y_real)
            
            self.window = window
            self.channel = channel


        return _ssl(y_pred, y_real, window, self.window_size, channel, self.beta, self.gamma, self.eps)

def ssl(y_pred, y_real, window_size = 11, beta=0.1, gamma=0.5, eps=1e-6):
    (_, channel, _, _) = y_real.size()
    window = create_window(window_size, channel)
    
    if y_real.is_cuda:
        window = window.cuda(y_real.get_device())
    window = window.type_as(y_real)
    
    return _ssl(y_pred, y_real, window, window_size, channel, beta, gamma, eps)