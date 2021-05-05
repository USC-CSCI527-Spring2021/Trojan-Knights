from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import h5py

import numpy as np
from tensorboardX import SummaryWriter

from models.autoencoder import AE

parser = argparse.ArgumentParser(description='Missing Pieces Method')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=5e-3, metavar='N',
                    help='learning rate (default: 5e-3)')
parser.add_argument('--decay', type=float, default=.98, metavar='N',
                    help='decay rate of learning rate (default: .98)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

lr = args.lr
decay = args.decay
batch_size=args.batch_size

writer = SummaryWriter(comment='lr: {} | decay: {} | batch size: {}'.format(lr, decay, batch_size))

class TrainSet(Dataset):
    def __init__(self, train_path):
        self.train_path = train_path

    def __getitem__(self, index):
        with h5py.File(self.train_path, 'r') as hf:
            data = hf['train'][index]
            x = data[:-1]
            y = data[-1]
            
            if y == 1:
                pieces_present_at = np.argwhere(x[:384] == 1)
                pieces_present_at = pieces_present_at.squeeze()
                removed_piece_idx = np.random.choice(pieces_present_at)
                x_piece_removed = x.copy()
                x_piece_removed[removed_piece_idx] = 0
            else:
                pieces_present_at = np.argwhere(x[384:768] == 1)
                pieces_present_at = pieces_present_at.squeeze()
                removed_piece_idx = np.random.choice(pieces_present_at)
                x_piece_removed = x.copy()
                x_piece_removed[removed_piece_idx] = 0                                              
            
            x_np = np.asarray(x, dtype=np.int32)
            y_np = np.asarray(y, dtype=np.int32)
            x_piece_removed_np = np.asarray(x_piece_removed, dtype=np.int32)
            
        return torch.from_numpy(x_np).type(torch.FloatTensor), torch.from_numpy(x_piece_removed_np).type(torch.FloatTensor), torch.from_numpy(y_np).type(torch.FloatTensor)

    def __len__(self):
        with h5py.File(self.train_path, 'r') as hf:
            return len(hf['train'])

class ValidSet(Dataset):
    def __init__(self, valid_path):
        self.valid_path = valid_path

    def __getitem__(self, index):
        with h5py.File(self.valid_path, 'r') as hf:
            data = hf['test'][index]
            x = data[:-1]
            y = data[-1]
            
            if y == 1:
                pieces_present_at = np.argwhere(x[:384] == 1)
                pieces_present_at = pieces_present_at.squeeze()
                removed_piece_idx = np.random.choice(pieces_present_at)
                x_piece_removed = x.copy()
                x_piece_removed[removed_piece_idx] = 0
            else:
                pieces_present_at = np.argwhere(x[384:768] == 1)
                pieces_present_at = pieces_present_at.squeeze()
                removed_piece_idx = np.random.choice(pieces_present_at)
                x_piece_removed = x.copy()
                x_piece_removed[removed_piece_idx] = 0    
            
            x_np = np.asarray(x, dtype=np.int32)
            y_np = np.asarray(y, dtype=np.int32)
            x_piece_removed_np = np.asarray(x_piece_removed, dtype=np.int32)
            
        return torch.from_numpy(x_np).type(torch.FloatTensor), torch.from_numpy(x_piece_removed_np).type(torch.FloatTensor), torch.from_numpy(y_np).type(torch.FloatTensor)

    def __len__(self):
        with h5py.File(self.valid_path, 'r') as hf:
            return len(hf['test'])


train_loader = torch.utils.data.DataLoader(TrainSet("/home/parthsuresh/DeepChess/data/train.h5"), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(ValidSet("/home/parthsuresh/DeepChess/data/test.h5"), batch_size=batch_size, shuffle=False)

model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def bce_loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 773), size_average=False)
    return BCE

def mse_loss_function(recon_x, x):
    MSE = F.mse_loss(recon_x, x.view(-1, 773), size_average=False)
    return MSE
        
def loss_function(recon_x, x, smooth=1):
    bce = bce_loss_function(recon_x, x)
    recon_x = recon_x.view(-1)
    x = x.view(-1)
    intersection = (recon_x * x).sum()                            
    dice = (2.*intersection + smooth)/(recon_x.sum() + x.sum() + smooth)      
    dice_loss = 1 - dice
    return dice_loss + bce
  

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x_og, x_noised, _) in enumerate(train_loader):
        x_og = x_og.to(device)
        x_noised = x_noised.to(device)
        optimizer.zero_grad()
        recon_batch, enc = model(x_noised)
        loss = loss_function(recon_batch, x_og)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_og), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(x_og)))
            writer.add_scalar('data/train_loss', loss.item() / len(x_og), epoch*len(train_loader) + batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_mse = 0
    total_diff = 0
    with torch.no_grad():
        for i, (x_og, x_noised, _) in enumerate(valid_loader):
            x_og = x_og.to(device)
            x_noised = x_noised.to(device)
            recon_batch, enc = model(x_noised)
            pred = (recon_batch.cpu().detach().numpy() > .5).astype(int)
            total_diff += float(np.sum(x_og.cpu().detach().numpy() != pred))
            test_loss += loss_function(recon_batch, x_og).item()
            test_loss_mse += mse_loss_function(recon_batch, x_og).item()

    test_loss /= len(valid_loader.dataset)
    test_loss_mse /= len(valid_loader.dataset)
    total_diff /= len(valid_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set loss (mse): {:.4f}'.format(test_loss_mse))
    print('====> Test set diff: {:.4f}'.format(total_diff))
    writer.add_scalar('data/test_loss', test_loss, epoch)
    writer.add_scalar('data/test_loss_mse', test_loss_mse, epoch)
    writer.add_scalar('data/test_diff', total_diff, epoch)

def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = 'checkpoints/self_sup/lr_{}_decay_{}'.format(int(lr*1000), int(decay*100))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'ae_{}.pth.tar'.format(epoch)))

def recon(game):
    recon, _ = model(torch.from_numpy(game).type(torch.FloatTensor))
    recon = (recon.cpu().detach().numpy() > .5).astype(int)
    return recon

start_epoch = 0
resume = False
if resume:
    state = torch.load('./checkpoints/best_autoencoder.pth.tar', 
                        map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']

for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    test(epoch)
    save(epoch)

    # Adjust learning rate
    for params in optimizer.param_groups:
        params['lr'] *= decay
