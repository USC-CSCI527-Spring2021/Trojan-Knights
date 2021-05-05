from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from models.autoencoder import AE

class RegNet(nn.Module):
    def __init__(self, load_weights=False):
        super(RegNet, self).__init__()

        self.encoder = AE()

        if load_weights:
            state = torch.load('checkpoints/best_autoencoder.pth.tar', map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(state['state_dict'])

        self.fc1 = nn.Linear(200, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 1)
    
    def forward_once(self, x):
        _, enc = self.encoder(x)
        return enc

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        x = torch.cat([x1, x2], dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

def loss_function(recon_x, x):
    ce = F.cross_entropy(recon_x, x, size_average=False)
    return ce
