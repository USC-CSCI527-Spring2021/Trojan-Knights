from __future__ import print_function
import argparse
import os
import random
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import h5py
from tensorboardX import SummaryWriter
from models.siamese_paper import Siamese
from models.autoencoder import AE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=.01, metavar='N',
                    help='learning rate (default: .01)')
parser.add_argument('--decay', type=int, default=.99, metavar='N',
                    help='decay rate of learning rate (default: .99)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

lr = args.lr
decay = args.decay
batch_size = args.batch_size

writer = SummaryWriter(comment='lr: {} | decay: {} | batch size: {}'.format(lr, decay, batch_size))

class TrainSet(Dataset):
    def __init__(self, train_path):
        self.train_path = train_path

    def __getitem__(self, index):
        with h5py.File(self.train_path, 'r') as hf:
            datalen = len(hf["train"])
            while True:
                random_win_idx = random.randint(0,datalen-1)
                random_win = hf["train"][random_win_idx]
                if random_win[-1] == 1: 
                    random_win = hf["train"][random_win_idx][:-1]
                    break
            while True:
                random_loss_idx = random.randint(0,datalen-1)
                random_loss = hf["train"][random_loss_idx]
                if random_loss[-1] == 0: 
                    random_loss = hf["train"][random_loss_idx][:-1]
                    break
            order = random.randint(0,1)
            if order == 0:
                x1, x2 = random_win, random_loss
                x1 = torch.from_numpy(x1).type(torch.FloatTensor)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor)
                label = torch.from_numpy(np.array(0)).type(torch.LongTensor)
            else:
                x1, x2 = random_loss, random_win
                x1 = torch.from_numpy(x1).type(torch.FloatTensor)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor)
                label = torch.from_numpy(np.array(1)).type(torch.LongTensor)
            return (x1, x2, label)

    def __len__(self):
        with h5py.File(self.train_path, 'r') as hf:
            return len(hf['train'])

class ValidSet(Dataset):
    def __init__(self, valid_path):
        self.valid_path = valid_path

    def __getitem__(self, index):
        with h5py.File(self.valid_path, 'r') as hf:
            datalen = len(hf["test"])
            while True:
                random_win_idx = random.randint(0,datalen-1)
                random_win = hf["test"][random_win_idx]
                if random_win[-1] == 1: 
                    random_win = hf["test"][random_win_idx][:-1]
                    break
            while True:
                random_loss_idx = random.randint(0,datalen-1)
                random_loss = hf["test"][random_loss_idx]
                if random_loss[-1] == 0: 
                    random_loss = hf["test"][random_loss_idx][:-1]
                    break
            order = random.randint(0,1)
            if order == 0:
                x1, x2 = random_win, random_loss
                x1 = torch.from_numpy(x1).type(torch.FloatTensor)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor)
                label = torch.from_numpy(np.array(0)).type(torch.LongTensor)
            else:
                x1, x2 = random_loss, random_win
                x1 = torch.from_numpy(x1).type(torch.FloatTensor)
                x2 = torch.from_numpy(x2).type(torch.FloatTensor)
                label = torch.from_numpy(np.array(1)).type(torch.LongTensor)
            return (x1, x2, label)

    def __len__(self):
        with h5py.File(self.valid_path, 'r') as hf:
            return len(hf['test'])

train_loader = torch.utils.data.DataLoader(TrainSet("/home/parthsuresh/DeepChess/data/train.h5"), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(ValidSet("/home/parthsuresh/DeepChess/data/test.h5"), batch_size=batch_size, shuffle=False)


# class TrainSet(Dataset):
#     def __init__(self, length):
#         self.length = length

#     def __getitem__(self, index):
#         rand_win = train_games_wins[
#             np.random.randint(0, train_games_wins.shape[0])]
#         rand_loss = train_games_losses[
#             np.random.randint(0, train_games_losses.shape[0])]

#         #rand_win = train_games_wins[0]
#         #rand_loss = train_games_losses[1234]

#         order = np.random.randint(0,2)
#         if order == 0:
#             stacked = np.hstack((rand_win, rand_loss))
#             stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
#             label = torch.from_numpy(np.array(0)).type(torch.LongTensor)
#             return (stacked, label)
#         else:
#             stacked = np.hstack((rand_loss, rand_win))
#             stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
#             label = torch.from_numpy(np.array(1)).type(torch.LongTensor)
#             return (stacked, label)

#     def __len__(self):
#         return self.length

# class TestSet(Dataset):
#     def __init__(self, length):
#         self.length = length

#     def __getitem__(self, index):
#         rand_win = test_games_wins[np.random.randint(0, test_games_wins.shape[0])]
#         rand_loss = test_games_losses[np.random.randint(0, test_games_losses.shape[0])]

#         order = np.random.randint(0,2)
#         if order == 0:
#             stacked = np.hstack((rand_win, rand_loss))
#             stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
#             label = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
#             return (stacked, label)
#         else:
#             stacked = np.hstack((rand_loss, rand_win))
#             stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
#             label = torch.from_numpy(np.array(1)).type(torch.FloatTensor)
#             return (stacked, label)

#     def __len__(self):
#         return self.length

# train_loader = torch.utils.data.DataLoader(TrainSet(5000000),batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(TestSet(200000),batch_size=batch_size, shuffle=True)


print('Buidling model...')
model = Siamese().to(device)
state = torch.load('./checkpoints/best_siamese.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
e = enumerate(test_loader)
b, (x1, x2, label) = next(e)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(pred, label):
    ce = F.cross_entropy(pred, label, size_average=False)
    return ce
 

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x1, x2, label) in enumerate(test_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)
            pred = model(x1, x2)
            test_loss += loss_function(pred, label).item()

            print(F.softmax(pred[:10], dim=1).cpu().detach().numpy(), label[:10].cpu().detach().numpy())


    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


print('Begin test...')
test(1)
