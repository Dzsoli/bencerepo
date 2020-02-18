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
from sklearn import metrics as m
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
    l_data = np.reshape(l_data, (-1, N, features, window_size))
    # train data
    train_data = np.reshape(l_data[0:int((1 - q) * l_data.shape[0])], (-1, features, window_size))
    train_labels = np.reshape(l_labels[0:int((1 - q) * l_data.shape[0])], (-1, 3))
    dataset_train = Trajectories(dataset=train_data, labels=train_labels, featnumb=features, window_size=window_size,
                                 transform=ToDevice())
    l_train_loader = DataLoader(dataset_train, batch_size=train_data.shape[0], shuffle=True)

    # validation data
    # valid_data = np.reshape(l_data[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])], (-1, 3, 30))
    # valid_labels = np.reshape(l_labels[int((1 - 2 * q) * l_data.shape[0]):int((1 - q) * l_data.shape[0])], (-1, 3))
    # dataset_valid = Trajectories(dataset=valid_data, labels=valid_labels, transform=ToDevice())
    # valid_loader = DataLoader(dataset_valid, batch_size=int(q * l_data.shape[0]) + 1, shuffle=True)

    # test data
    test_data = np.reshape(l_data[int((1 - q) * l_data.shape[0]):], (-1, features, window_size))
    test_labels = np.reshape(l_labels[int((1 - q) * l_data.shape[0]):], (-1, 3))
    dataset_test = Trajectories(dataset=test_data, labels=test_labels, featnumb=features, window_size=window_size,
                                transform=ToDevice())
    l_test_loader = DataLoader(dataset_test, batch_size=test_data.shape[0], shuffle=True)

    return l_train_loader, l_test_loader


def run(l_test_loader, l_train_loader, l_model, l_loss_fn, l_optimizer, l_num_epochs, l_lr):

    path = "../../../results"
    los = []
    acc = []
    f1_scores = []
    val_error = []

    print("device is: ", models.device)
    for epoch in range(l_num_epochs):
        l_model.train()
        if epoch % 10 == 0:
            print("Epoch: ", epoch)
        for i, sample in enumerate(l_train_loader):
            out = l_model(sample['data'])
            out = out.to(models.device)
            loss = l_loss_fn(out, sample['label'])

            l_optimizer.zero_grad()
            loss.backward()
            l_optimizer.step()

            los.append(loss)

        val_loss, val_corr, val_total, f1_score = valid(l_model, l_test_loader, l_loss_fn)
        val_acc = val_corr / val_total
        acc.append(val_acc)
        val_error.append(val_loss)
        f1_scores.append(f1_score)

    directory = path + "/" + str(l_lr)
    if not os.path.exists(directory):
        os.makedirs(directory)

    values = np.array([los, val_error, acc]).transpose((1, 0))
    pd.DataFrame([los, val_error, acc]).to_csv(directory + '/losses.csv')
    pd.DataFrame(f1_scores).to_csv(directory + '/f1_scores.csv')
    fig = plt.figure()
    plt.plot(los)
    plt.ylabel('Training loss')
    plt.ylabel('Epoch')
    fig.savefig(directory + '/training_loss.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(val_error)
    plt.ylabel('Validating loss')
    plt.xlabel('Epoch')
    fig.savefig(directory + '/valid_loss.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(acc)
    plt.ylabel('Validating accuracy')
    plt.xlabel('Epoch')
    fig.savefig(directory + '/valid_acc.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(f1_scores)
    plt.ylabel('F scores')
    plt.xlabel('Epoch')
    fig.savefig(directory + '/f1_scores.png')
    del fig
    # plt.show()

    fig = plt.figure()
    plt.plot(values)
    plt.ylabel('Training and validation loss, and accuracy')
    plt.xlabel('Epoch')
    plt.axes().set_ylim(ymax=1.0)
    fig.savefig(directory + '/training_losses.png')
    del fig


def valid(l_model, l_valid_loader, l_loss_fn):
    l_model.eval()
    correct = 0
    total = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0
    good = 0
    bad = 0

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
                else:
                    bad = bad + 1

            # f_1_score = f_measure(true_pos, false_pos, false_neg, 1.)
            # f_2_score = f_measure(true_pos, false_pos, false_neg, 2.)
            # f_05_score = f_measure(true_pos, false_pos, false_neg, 0.5)
            # recall = true_pos / (true_pos + false_neg)
            # precision = true_pos / (true_pos + false_pos)
            sk_f1_score = m.f1_score(np.array(lab_idx.cpu()), np.array(out_idx.cpu()), average=None)
            # print("RECALL: ", recall, "PRECISION: ", precision, "F1: ", 2 * recall * precision / (recall + precision),
            #       "ACCURACY: ", good / (good + bad), "SKLEARN_F_1: ", sk_f1_score)
            # print('C= ', C, 'TP= ', true_pos, 'FP= ', false_pos, 'FN= ', false_neg)
            # TODO: tensor element wise product

        return loss, correct, total, sk_f1_score


if __name__ == '__main__':
    lr = 0.0501
    num_epochs = 1000
    # batch_size = 512

    train_loader, test_loader = load_data('dX_Y')
    input_dim = train_loader.dataset.featnumb

    # model and optimizer
    model = SimpleLSTM(input_dim, 7, 1).to(models.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)

    run(l_train_loader=train_loader, l_test_loader=test_loader, l_model=model, l_loss_fn=loss_fn,
        l_num_epochs=num_epochs, l_optimizer=optimizer_adam, l_lr=lr)
