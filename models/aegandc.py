from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from utils.dataset_utils import oriDataset

class aegenerator(nn.Module):
    def __init__(self):
        super(aegenerator, self).__init__()
        self.convblock = nn.Sequential(
            # input image channel 224 x 224 x 3
            nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #112
            nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #56
            nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),         
            nn.MaxPool2d(kernel_size=2, stride=2),  
            #28
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #14
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(True)


        )
        self.ctblock = nn.Sequential(
            #256x56x14
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #128x112x28
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #56
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #112
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            #224

            nn.Tanh()
        )

    def forward(self, input):
        convoutput = self.convblock(input)
        print(convoutput.shape)
        output = self.ctblock(convoutput)
        return output
if __name__ == "__main__":
    aegenerator()

