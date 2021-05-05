import h5py
import random
import gc
import tables as tb
from tqdm import tqdm
import numpy as np


filetr = tb.open_file('data/train.h5', 'w')
filets = tb.open_file('data/test.h5', 'w')

hf = h5py.File('data/games.h5', 'r')

"""
ww_idx = []
bw_idx = []

print("Collecting indices...")
for i in tqdm(range(len(hf["games"]))):
    if hf["games"][i][-1] == 1:
        ww_idx.append(i)
    else:
        bw_idx.append(i)

random.shuffle(ww_idx)
random.shuffle(bw_idx)

train_idxs = ww_idx[:2500000] + bw_idx[:2500000]
train_idxs = sorted(train_idxs)
print("Train idx len : ", len(train_idxs))

test_idxs = ww_idx[2500000:2600000] + bw_idx[2500000:2600000]
test_idxs = sorted(test_idxs)
print("Test idx len : ", len(test_idxs))

np.save('train_idxs.npy', np.array(train_idxs))
np.save('test_idxs.npy', np.array(test_idxs))

print("Completed index collection")
"""

train_idxs = np.load('train_idxs.npy')
test_idxs = np.load('test_idxs.npy')

print("Test write started...")
game_example = np.expand_dims(hf["games"][test_idxs[0]], axis=0)
test_arr = filets.create_earray( filets.root, "test", obj=game_example )
for i in tqdm(range(1, len(test_idxs))):
    game = np.expand_dims(hf["games"][test_idxs[i]], axis=0)
    test_arr.append(game)
filets.close()
print("Test write completed")

print("Training write started...")
game_example = np.expand_dims(hf["games"][train_idxs[0]], axis=0)
train_arr = filetr.create_earray( filetr.root, "train", obj=game_example )
for i in tqdm(range(1, len(train_idxs))):
    game = np.expand_dims(hf["games"][train_idxs[i]], axis=0)
    train_arr.append(game)
filetr.close()
print("Training write completed")

hf.close()
