import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import relu, log_softmax
from itertools import product
from functools import reduce


class FacialDataset(Dataset):
    def __init__(self, datapath, conv=False):
        super().__init__()
        self.data = pd.read_csv(datapath)       # Load data
        self.data = self.data.dropna()          # Drop NaN containing rows

        self.images = self.data['Image'].apply(lambda im: np.fromstring(im, sep=' ')).values    # Extract images as nda
        self.images = self.images / 255
        if conv:
            self.images = np.stack(self.images).reshape(-1, 1, 96, 96)

        self.annots = self.data.values[:, :-1].astype(float)
        self.annots = (self.annots - 96/2) / (96/2)



    def __getitem__(self, idx):
        return self.images[idx], self.annots[idx]

    def __len__(self):
        return len(self.annots)


class FacialNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96*96, 100)
        self.fc2 = nn.Linear(100, 30)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.fc1(input)
        x = self.activation(x)
        out = self.fc2(x)
        return out


class FacialNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.fc1 = nn.Linear(15488, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 30)

        self.activation = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.mp(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.mp(x)
        x = self.activation(x)

        x = x.view(-1, 15488)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        out = self.fc3(x)
        return out
