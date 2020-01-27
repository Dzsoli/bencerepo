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


def load_data(features: str, path='../../../full_data/'):

    l_data = np.load(path + features + '_dataset.npy')
    l_labels = np.load(path + features + '_labels.npy')

    # quotient for chopping the data
    q = 0.2

    # TODO: the window_size and shift (and the N=6) is not in this scope
    window_size = l_data.shape[3]
    features = l_data.shape[2]
    N = l_data.shape[1]
    l_data = np.reshape(l_data, (-1, features, window_size))

    # normalize
    l_data = l_data / np.linalg.norm(l_data, axis=0)
    l_data = np.reshape(l_data, (-1, 6, features, window_size))
    # train data
    train_data = torch.from_numpy(np.reshape(l_data[0:int((1 - q) * l_data.shape[0])], (-1, features, window_size))).float()
    train_labels = torch.from_numpy(np.reshape(l_labels[0:int((1 - q) * l_data.shape[0])], (-1, 3))).float()
    dataset_train = TensorDataset(train_data, train_labels)
    l_train_loader = DataLoader(dataset_train, batch_size=train_data.shape[0], shuffle=True)

    # validation data
    # valid_data = np.reshape(l_data[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])], (-1, 3, 30))
    # valid_labels = np.reshape(l_labels[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])], (-1, 3))
    # dataset_valid = Trajectories(dataset=valid_data, labels=valid_labels, transform=ToDevice())
    # valid_loader = DataLoader(dataset_valid, batch_size=int(q * l_data.shape[0]) + 1, shuffle=True)

    # test data
    test_data = torch.from_numpy(np.reshape(l_data[int((1 - q) * l_data.shape[0]):], (-1, features, window_size))).float()
    test_labels = torch.from_numpy(np.reshape(l_labels[int((1 - q) * l_data.shape[0]):], (-1, 3))).float()
    dataset_test = TensorDataset(test_data, test_labels)
    l_test_loader = DataLoader(dataset_test, batch_size=test_data.shape[0], shuffle=True)

    return l_train_loader, l_test_loader


if __name__ == '__main__':
    lr = 0.05
    num_epochs = 750
    # batch_size = 512
    train_loader, test_loader = load_data('dX_V_A')
    input_dim = train_loader.dataset.tensors[0].shape[1]
    # model and optimizer
    model = CNN(input_dim).to(models.device)
    loss_fn = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # if use_cuda:
    #     x_test = x_test.cuda()
    #     y_test = y_test.cuda()
    #     model = model.cuda()

    los = []
    acc = []
    val_error = []

    for epoch in range(num_epochs):
        model.train()
        # tic = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            # print(inputs)

            inputs, labels = inputs.cuda(), labels.cuda()

            preds = model(inputs)
            preds = preds.cuda()
            # print(labels.size())
            # print(preds.size())
            # print(preds)
            # print(labels)

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            los.append(loss)

        # val_acc, val_err = testing(model, valid_loader)
        # acc.append(val_acc)
        # val_error.append(val_err)

        model.eval()
        # print('[epoch: {:d}] train_loss: {:.3f}, ({:.1f}s)'.format(epoch, loss.item(), time.time()-tic) )  # pytorch 0.4 and later

    plt.plot(los)
    plt.ylabel('Training loss')
    plt.show()

    # plt.plot(val_error)
    # plt.ylabel('Validating loss')
    # plt.show()
    #
    # plt.plot(acc)
    # plt.ylabel('Validating accuracy')
    # plt.show()