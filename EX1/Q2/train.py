import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import IrisDataset, IrisNet

parser = argparse.ArgumentParser()
parser.add_argument('--dpath', type=str, default='iris.data.csv', help='length of a bit sequence')
parser.add_argument('--epochs', type=int, default=750, help='Number of epochs to train on')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--no_gpu', action='store_true', help='Disable gpu usage')


def run(device):
    net = IrisNet()
    net.to(device)

    optimizer = Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_set = IrisDataset(args.dpath)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    losses = []
    accuracy = []
    # Begin training
    for ep in range(args.epochs):
        ep_loss = 0
        ep_acc = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, probs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()*len(inputs)
            ep_acc += sum(probs.argmax(dim=-1) == labels).item()
        losses.append(ep_loss/len(train_set))
        accuracy.append(ep_acc/len(train_set))

    return net, losses, accuracy


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu and torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    _, loss, acc = run(dev)

    print(f'Fully Connected Network On Iris Dataset')
    print(f'Final training loss: {loss[-1]}')
    print(f'Final Accuracy: {acc[-1]*100}%')

    # Plot results
    plt.figure()
    plt.plot(range(args.epochs), loss)
    plt.title('Training Loss - CrossEntropyLoss vs Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('Loss.png')

    plt.figure()
    plt.plot(range(args.epochs), acc)
    plt.title('Training Accuracy vs Epoch')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.savefig('Accuracy.png')



