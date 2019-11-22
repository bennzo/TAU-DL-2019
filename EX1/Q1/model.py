import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import product
from functools import reduce


class XORDataset(Dataset):
    def __init__(self, nbits=2):
        super().__init__()
        self.bits = np.array(list(product([0, 1], repeat=nbits)))
        self.labels = [reduce(lambda i, j: i ^ j, bit_seq) for bit_seq in self.bits]

    def __getitem__(self, idx):
        return self.bits[idx], self.labels[idx]

    def __len__(self):
        return len(self.bits)


class XORNet(nn.Module):
    def __init__(self, nbits=2, hsize=3):
        super().__init__()
        self.in_layer = nn.Linear(nbits, hsize)
        self.hidden_layer = nn.Linear(hsize, hsize)
        self.out_layer = nn.Linear(hsize, 1)
        self.activation = nn.Tanh()

    def forward(self, bit_seq):
        x = self.in_layer(bit_seq)
        x = self.hidden_layer(x)
        x = self.activation(x)
        out = self.out_layer(x)
        return out.flatten()
