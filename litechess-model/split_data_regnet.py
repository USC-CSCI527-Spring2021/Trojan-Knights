import h5py
import random
import gc
import tables as tb
from tqdm import tqdm
import numpy as np

filetr = tb.open_file('data/train_reg.h5', 'w')
filets = tb.open_file('data/test_reg.h5', 'w')

hf = h5py.File('data/games_reg.h5', 'r')

idxs = list(range(12958035))
random.shuffle(idxs)
train_idxs = idxs[:int(0.8 * len(idxs))]
test_idxs = idxs[int(0.8 * len(idxs)): ]

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
