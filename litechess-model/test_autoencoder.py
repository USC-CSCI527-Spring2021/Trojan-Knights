from models.autoencoder import AE
import numpy as np
import torch
import h5py
import tables as tb
from tqdm import tqdm

model = AE()
state = torch.load('checkpoints/best_autoencoder.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])

with h5py.File('data/train.h5', 'r') as hf:
    data = hf["train"][:10]
    game, winner = data[:, :-1], data[:, -1]
    recon, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    print(list(zip(game, recon.detach().numpy())))