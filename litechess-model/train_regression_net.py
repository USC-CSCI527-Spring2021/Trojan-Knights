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
from models.regnet import RegNet
from models.autoencoder import AE

parser = argparse.ArgumentParser(description='Evaluation Score Regression')
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
            datalen = len(hf['train'])
            idx1 = random.randint(0, datalen-1)
            idx2 = random.randint(0, datalen-1)
            x1 = hf['train'][idx1,:-1]
            x2 = hf['train'][idx2,:-1]
            label1 = hf['train'][idx1,-1]
            label2 = hf['train'][idx2,-1]
            label = abs(label1 - label2) / 20000
            x1 = torch.from_numpy(x1).type(torch.FloatTensor)
            x2 = torch.from_numpy(x2).type(torch.FloatTensor)
            label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
            return (x1, x2, label)

    def __len__(self):
        with h5py.File(self.train_path, 'r') as hf:
            return len(hf['train'])

class ValidSet(Dataset):
    def __init__(self, valid_path):
        self.valid_path = valid_path

    def __getitem__(self, index):
        with h5py.File(self.valid_path, 'r') as hf:
            datalen = len(hf['test'])
            idx1 = random.randint(0, datalen-1)
            idx2 = random.randint(0, datalen-1)
            x1 = hf['test'][idx1,:-1]
            x2 = hf['test'][idx2,:-1]
            label1 = hf['test'][idx1,-1]
            label2 = hf['test'][idx2,-1]
            label = abs(label1 - label2) / 20000
            x1 = torch.from_numpy(x1).type(torch.FloatTensor)
            x2 = torch.from_numpy(x2).type(torch.FloatTensor)
            label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
            return (x1, x2, label)

    def __len__(self):
        with h5py.File(self.valid_path, 'r') as hf:
            return len(hf['test'])

train_loader = torch.utils.data.DataLoader(TrainSet("/home/parthsuresh/DeepChess/data/train_reg.h5"), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(ValidSet("/home/parthsuresh/DeepChess/data/test_reg.h5"), batch_size=batch_size, shuffle=False)

print('Buidling model...')
model = RegNet(load_weights=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

e = enumerate(train_loader)
b, (x1, x2, label) = next(e)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(pred, label):
    mse = F.mse_loss(pred, label, size_average=False)
    return mse


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x1, x2,label) in enumerate(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(x1, x2)
        pred_label = pred.argmax(dim=1)
        loss = loss_function(pred, label)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(x1)))
            print(pred[:10].cpu().detach().numpy(), label[:10].cpu().detach().numpy())
            #idx = random.randint(0,127)
            #print(F.softmax(pred[idx]).cpu().detach().numpy(), label[idx].cpu().detach().numpy())
            writer.add_scalar('data/train_loss', loss.item() / len(x1), epoch*len(train_loader) + batch_idx)
        
        

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    

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

            if i % args.log_interval == 0:
                print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                (i+1) * len(x1), len(test_loader.dataset),
                100. * (i+1) / len(test_loader),
                test_loss / ((i+1) * len(x1))))


    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('data/test_loss', test_loss, epoch)

def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = 'checkpoints/reg_net/lr_{}_decay_{}'.format(int(lr*1000), int(decay*100))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'regnet_{}.pth.tar'.format(epoch)))

start_epoch = 1
resume = False
if resume:
    state = torch.load('./checkpoints/sm_4.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    #optimizer = optim.SGD(model.parameters(), lr=3e-4)
    start_epoch = state['epoch']

print('Begin train...')
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    test(epoch)
    save(epoch)

    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
