import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import relu


class IrisDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.iris_feats = np.genfromtxt(datapath, delimiter=',', dtype=float, usecols=range(0,4))
        self.iris_specs = np.genfromtxt(datapath, delimiter=',', dtype=str, usecols=[4])

        uniq_specs = np.unique(self.iris_specs)
        spec_idx = dict(zip(uniq_specs, range(len(uniq_specs))))
        self.labels = list(map(lambda spec: spec_idx[spec], self.iris_specs))

    def __getitem__(self, idx):
        return self.iris_feats[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class IrisNet(nn.Module):
    def __init__(self, insize=4, hsize=10, outsize=3):
        super().__init__()
        self.fc1 = nn.Linear(insize, hsize)
        self.fc2 = nn.Linear(hsize, outsize)
        self.activation = ReQU()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.fc1(input)
        x = self.activation(x)
        out = self.fc2(x)
        probs = self.sm(out)
        return out, probs


class ReQU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)*relu(x)
