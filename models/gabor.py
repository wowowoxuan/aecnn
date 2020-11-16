import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import math
import sys
import numpy as np
import scipy.misc
import time
from PIL import Image
import cv2
def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in = 3, channel_out = 3, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = 7
        self.channel_in = 3
        self.channel_out = 3
        self.stride = stride
        self.padding = padding

        self.Lambda = torch.FloatTensor([np.pi/2])
        self.theta = torch.FloatTensor([np.pi/3])
        self.psi = torch.FloatTensor([np.pi/2])
        self.sigma = torch.FloatTensor([1])
        self.gamma = torch.FloatTensor([0.5])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.theta
        gamma = self.gamma
        sigma = self.sigma
        Lambda = self.Lambda
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float().cuda()   # [channel_out, channel_in, kernel, kernel]

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out
