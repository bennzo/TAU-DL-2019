import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
from model import XORDataset, XORNet

parser = argparse.ArgumentParser()
parser.add_argument('--nbits', type=int, default='2', help='length of a bit sequence')
parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs to train on')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--no_gpu', action='store_true', help='Disable gpu usage')


def run(device):
    net = XORNet(args.nbits)
    net.to(device)

    optimizer = Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    train_set = XORDataset(args.nbits)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)

    losses = []
    # Begin training
    for ep in range(args.epochs):
        ep_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()*len(inputs)
        losses.append(ep_loss/len(train_set))

    # Test trained network
    outputs = []
    for (inputs, labels) in train_loader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        outputs.append(net(inputs))

    # Print results
    print(f'MSE Loss: {losses[-1]}')
    print('Predictions:')
    print(f'Input | True XOR | Prediction')
    print('\n'.join([f'{bits} | {label} | {out}' for bits, out, label in zip(train_set.bits, torch.cat(outputs), train_set.labels)]))
    return net, losses


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu and torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    run(dev)
