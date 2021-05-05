from models.autoencoder import AE
import numpy as np
import torch
import h5py
import tables as tb
from tqdm import tqdm

model = AE()
state = torch.load('checkpoints/best_autoencoder.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])

def featurize(game):
    model.eval()
    recon, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

fileh = tb.open_file('data/enc_train.h5', mode='w')
filel = tb.open_file('data/enc_train_eval.h5', mode='w')

filej = tb.open_file('data/enc_test.h5', mode='w')
filek = tb.open_file('data/enc_test_eval.h5', mode='w')


with h5py.File('data/train.h5', 'r') as hf:
    train_len = len(hf["train"])
    chunk_size = 1000
    
    data = hf["train"][0:chunk_size]
    game, winner = data[:, :-1], data[:, -1]
    feat = featurize(game)
    games_arr = fileh.create_earray( fileh.root, "enc_train", obj=feat )
    labels_arr = filel.create_earray( filel.root, "enc_train_labels", obj=winner )

    for idx in tqdm(range(1, (train_len//chunk_size))):
        data = hf["train"][idx*chunk_size: (idx+1)*chunk_size]
        game, winner = data[:, :-1], data[:, -1]
        feat = featurize(game)
        games_arr.append(feat)
        labels_arr.append(winner)

with h5py.File('data/test.h5', 'r') as hf:
    test_len = len(hf["test"])
    chunk_size = 1000
    
    data = hf["test"][0:chunk_size]
    game, winner = data[:, :-1], data[:, -1]
    feat = featurize(game)
    games_arr = filej.create_earray( filej.root, "enc_test", obj=feat )
    labels_arr = filek.create_earray( filek.root, "enc_test_labels", obj=winner )

    for idx in tqdm(range(1, (test_len//chunk_size))):
        data = hf["test"][idx*chunk_size: (idx+1)*chunk_size]
        game, winner = data[:, :-1], data[:, -1]
        feat = featurize(game)
        games_arr.append(feat)
        labels_arr.append(winner)
