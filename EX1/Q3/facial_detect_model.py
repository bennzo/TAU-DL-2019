import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

lr = 1e-4 # learning rate
bs = 16 # batch size
n_epoch = 100 # number of epochs

def train_model(x, y, x_valid, y_valid):
    model = nn.Sequential(
        nn.Linear(9216, 100), 
        nn.ReLU(), 
        nn.Linear(100, 30),
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_batch = math.ceil(len(x) / bs)

    loss_vals = []
    test_loss_vals = []

    for t in range(n_epoch):
        loss_val = 0
        for x_vecs, y_vecs in get_next_batch(x, y):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_vecs)
            # Compute loss
            loss = loss_fn(y_pred, y_vecs) # return the average
            loss_val += loss.item()
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        loss_val /= num_batch
        loss_vals.append(loss_val)
        test_loss_val = evaluate(model, loss_fn, x_valid, y_valid)
        test_loss_vals.append(test_loss_val)
        print(t, loss_val, test_loss_val)

    t_vals = np.arange(1, n_epoch + 1)

    plt.plot(t_vals, loss_vals, 'r--', label='train')
    plt.plot(t_vals, test_loss_vals, 'b--', label='test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper center', fontsize='x-large')
    plt.savefig('facial_detect_loss')
    plt.show()

def get_next_batch(x, y):
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
            yield torch.stack(x_vecs), torch.stack(y_vecs)
            x_vecs = []
            y_vecs = []

def evaluate(model, loss_fn, x, y):
    num_batch = math.ceil(len(x) / bs)
    loss_val = 0
    for x_vecs, y_vecs in get_next_batch(x, y):
        y_pred = model(x_vecs)
        loss = loss_fn(y_pred, y_vecs)
        loss_val += loss.item()
    return loss_val / num_batch