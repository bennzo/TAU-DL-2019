import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

lr = 1e-4 # learning rate
bs = 128 # batch size
n_epoch = 400 # number of epochs
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3), # 32 * 94 * 94
        nn.MaxPool2d(2, stride=2), # 32 * 47 * 47
        nn.ReLU(),
        nn.Conv2d(32, 64, 2), # 64 * 46 * 46
        nn.MaxPool2d(2, stride=2), # 64 * 23 * 23
        nn.ReLU(),
        nn.Conv2d(64, 128, 2), # 128 * 22 * 22
        nn.MaxPool2d(2, stride=2), # 128 * 11 * 11
        nn.ReLU(),
        nn.Flatten(), # 15488
        nn.Linear(15488, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 30)
    )
    return model

def create_simple_model():
    model = nn.Sequential(
        nn.Linear(9216, 100), 
        nn.ReLU(), 
        nn.Linear(100, 30),
    )
    return model

def train_model(x, y, x_valid, y_valid, model_name):
    print("Building {} model".format(model_name))

    if model_name == 'simple':
        model = create_simple_model()
    else:
        model = create_conv_model()

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_vals = []
    test_loss_vals = []

    for t in range(n_epoch):
        loss_val = 0
        for x_vecs, y_vecs in get_next_batch(x, y, model_name):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_vecs.to(device))
            # Compute loss
            loss = loss_fn(y_pred.to(device), y_vecs.to(device)) # return the average
            loss_val += loss.item()
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        loss_val /= len(x)
        loss_vals.append(loss_val)
        test_loss_val = evaluate(model, loss_fn, x_valid, y_valid, model_name)
        test_loss_vals.append(test_loss_val)
        print(t, loss_val, test_loss_val)

    t_vals = np.arange(1, n_epoch + 1)

    plt.plot(t_vals, loss_vals, 'r--', linewidth=3, label='train')
    plt.plot(t_vals, test_loss_vals, 'b--', linewidth=3, label='test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(loc='upper center', fontsize='x-large')
    fig_name = 'facial_detect_loss_'
    fig_name += model_name
    fig_name += '_model'
    plt.savefig(fig_name)
    plt.show()

def get_next_batch(x, y, model_name):
    x_vecs = []
    y_vecs = []

    n_inputs = len(x)
    rand_idx = np.arange(n_inputs)
    np.random.shuffle(rand_idx)

    for i in range(1, n_inputs + 1):
        ind = rand_idx[i - 1]
        x_vecs.append(x[ind])
        y_vecs.append(y[ind])
        if i % bs == 0 or i == n_inputs:
            x_vecs = torch.stack(x_vecs)
            y_vecs = torch.stack(y_vecs)
            if model_name == 'conv':
                x_vecs = x_vecs.view(len(x_vecs), 1, 96, 96)
            yield x_vecs, y_vecs
            x_vecs = []
            y_vecs = []

def evaluate(model, loss_fn, x, y, model_name):
    num_batch = math.ceil(len(x) / bs)
    loss_val = 0
    for x_vecs, y_vecs in get_next_batch(x, y, model_name):
        y_pred = model(x_vecs.to(device))
        loss = loss_fn(y_pred, y_vecs)
        loss_val += loss.item()
    return loss_val / len(x)