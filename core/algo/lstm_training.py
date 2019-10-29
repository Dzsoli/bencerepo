from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from core.utils import *
from core import *

import warnings
warnings.filterwarnings("ignore")


def run(l_data, l_labels, l_model, l_loss_fn, l_optimizer, l_batch_size='full'):

    # quotient for chopping the data
    q = 0.1

    # full batch:
    if l_batch_size == 'full':
        l_batch_size = int((1 - 2 * q) * l_data.shape[0])

    # normalize
    l_data = l_data / np.linalg.norm(l_data, axis=0)
    # train data
    train_data = l_data[0:int((1 - 2 * q) * l_data.shape[0])]
    train_labels = l_labels[0:int((1 - 2 * q) * l_data.shape[0])]
    dataset_train = Trajectories(dataset=train_data, labels=train_labels, transform=ToDevice())
    train_loader = DataLoader(dataset_train, batch_size=l_batch_size, shuffle=True)

    # validation data
    valid_data = l_data[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])]
    valid_labels = l_labels[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])]
    dataset_valid = Trajectories(dataset=valid_data, labels=valid_labels, transform=ToDevice())
    valid_loader = DataLoader(dataset_valid, batch_size=int(q * l_data.shape[0]) + 1, shuffle=True)

    # test data
    test_data = l_data[int((1 - q) * l_data.shape[0]):]
    test_labels = l_labels[int((1 - q) * l_data.shape[0]):]
    dataset_test = Trajectories(dataset=test_data, labels=test_labels, transform=ToDevice())
    test_loader = DataLoader(dataset_test, batch_size=int(q * l_data.shape[0]) + 1, shuffle=True)

    los = []
    acc = []
    val_error = []

    print("device is: ", models.device)
    for epoch in range(num_epochs):
        l_model.train()

        for i, sample in enumerate(train_loader):
            out = l_model(sample['data'])
            loss = l_loss_fn(out, sample['label'])

            l_optimizer.zero_grad()
            loss.backward()
            l_optimizer.step()

            los.append(loss)

        val_loss, val_corr, val_total = valid(l_model, valid_loader, loss_fn)
        val_acc = val_corr / val_total
        acc.append(val_acc)
        val_error.append(val_loss)

    fig = plt.figure()
    plt.plot(los)
    plt.ylabel('Training loss')
    fig.savefig('training_loss.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(val_error)
    plt.ylabel('Validating loss')
    fig.savefig('valid_loss.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(acc)
    plt.ylabel('Validating accuracy')
    fig.savefig('valid_acc.png')
    del fig
    # plt.show()


def valid(l_model, l_valid_loader, l_loss_fn):
    l_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, sample in enumerate(l_valid_loader):
            out = l_model(sample['data'])
            loss = l_loss_fn(out, sample['label'])

            out_idx = torch.argmax(out, 1)
            lab_idx = torch.argmax(sample['label'], 1)

            total = len(out_idx)

            for k in range(total):
                # total = total + 1
                if out_idx[k] == lab_idx[k]:
                    correct = correct + 1

            # TODO: tensor element wise product

        return loss, correct, total


if __name__ == '__main__':
    lr = 0.2
    num_epochs = 5000
    # batch_size = 512
    data = np.load('../../../full_data/dataset.npy')
    labels = np.load('../../../full_data/labels.npy')

    # model and optimizer
    model = SimpleLSTM(3, 7).to(models.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    run(l_data=data, l_labels=labels, l_model=model, l_loss_fn=loss_fn, l_optimizer=optimizer_adam)
