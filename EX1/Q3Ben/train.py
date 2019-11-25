import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from model import FacialDataset, FacialNet2, FacialNet3

parser = argparse.ArgumentParser()
parser.add_argument('--question', type=int, default=2, help='Question section')
parser.add_argument('--dpath', type=str, default='training.csv', help='length of a bit sequence')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train on')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--no_gpu', action='store_true', help='Disable gpu usage')


def run(device, question):
    if question == 2:
        net = FacialNet2()
        dataset = FacialDataset(args.dpath, conv=False)
    else:
        net = FacialNet3()
        dataset = FacialDataset(args.dpath, conv=True)
    net.to(device)

    optimizer = Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    idxs, split = list(range(len(dataset))), round(0.8*len(dataset))
    random.shuffle(idxs)
    train_set = Subset(dataset, idxs[:split])
    test_set = Subset(dataset, idxs[split:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    train_loss = []
    test_loss = []
    for ep in range(args.epochs):
        ep_train_loss = 0
        ep_test_loss = 0
        # Train loop
        net.train()
        for i, (images, annots) in enumerate(train_loader):
            images = images.float().to(device)
            annots = annots.float().to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, annots)
            loss.backward()
            optimizer.step()

            ep_train_loss += loss.item()*len(images)
        train_loss.append(ep_train_loss/len(train_set))

        # Test loop
        net.eval()
        for i, (images, annots) in enumerate(test_loader):
            images = images.float().to(device)
            annots = annots.float().to(device)

            outputs = net(images)
            loss = criterion(outputs, annots)

            ep_test_loss += loss.item()*len(images)
        test_loss.append(ep_test_loss/len(test_set))

    return net, train_loss, test_loss


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu and torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    random.seed(0)

    # Fully Connected Run
    _, train_loss, test_loss = run(dev, 2)
    print(f'Fully Connected Network on Facial Keypoint Detection:')
    print(f'Final training loss: {train_loss[-1]}')
    print(f'Final test loss: {test_loss[-1]}')
    # Plot results
    plt.figure()
    plt.plot(range(args.epochs), train_loss, label='train')
    plt.plot(range(args.epochs), test_loss, label='test')
    plt.title('{Training,Test} FCNet MSELoss vs Epoch\n(80-20 split)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('FC.png')

    # Convolutional Network Run
    _, train_loss, test_loss = run(dev, 3)
    print(f'Convolutional Network on Facial Keypoint Detection:')
    print(f'Final training loss: {train_loss[-1]}')
    print(f'Final test loss: {test_loss[-1]}')
    # Plot results
    plt.figure()
    plt.plot(range(args.epochs), train_loss, label='train')
    plt.plot(range(args.epochs), test_loss, label='test')
    plt.title('{Training,Test} ConvNet MSELoss vs Epoch\n(80-20 split)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Conv.png')
