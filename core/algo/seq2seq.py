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
import random
import math
import time
from core.utils import *
from core import common

import warnings
warnings.filterwarnings("ignore")


SEED = 420 + 911

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_data(features: str, path='../../../full_data/'):

    l_data = np.load(path + features + '_dataset.npy')
    # l_labels = np.load(path + features + '_labels.npy')
    # quotient for chopping the data
    q = 0.1

    # TODO: the window_size and shift (and the N=6) is not in this scope

    window_size = l_data.shape[3]
    features = l_data.shape[2]
    N = l_data.shape[1]
    l_data = np.reshape(l_data, (-1, features, window_size))

    # normalize
    l_data = l_data / np.linalg.norm(l_data, axis=0)
    l_data = np.reshape(l_data, (-1, N, features, window_size))
    # train data
    # [number of sequences, number of features, length of sequence]
    V = l_data.shape[0]
    train_data = np.reshape(l_data[0:int((1 - 2*q) * V)], (-1, features, window_size))
    test_data = np.reshape(l_data[int((1 - 2*q) * V):int(1 - q) * V], (-1, features, window_size))
    valid_data = np.reshape(l_data[int((1 - q) * V):], (-1, features, window_size))
    # train_labels = np.reshape(l_labels[0:int((1 - q) * l_data.shape[0])], (-1, 3))
    return N, features, window_size, train_data, test_data, valid_data


def count_parameters(l_model):
    return sum(p.numel() for p in l_model.parameters() if p.requires_grad)


def train(l_model, l_train_data, l_optimizer, l_criterion, l_clip):
    l_model.train()

    epoch_loss = 0

    src = l_train_data
    trg = l_train_data

    l_optimizer.zero_grad()

    output = l_model(src, trg)

    # trg = [trg len, batch size]
    # output = [trg len, batch size, output dim]
    # todo: reverse the output tensor
    for i in range(output.shape[1]):

        loss = l_criterion(output[:, i, :], trg[:, i, :])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(l_model.parameters(), l_clip)

        l_optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss


def evaluate(model, valid_data, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        src = valid_data
        trg = valid_data

        output = model(src, trg, 0)  # turn off teacher forcing

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        for i in range(output.shape[1]):
            loss = criterion(output[:, i, :], trg[:, i, :])
            epoch_loss += loss.item()

    return epoch_loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run():
    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    N, feature_dim, seq_length, train_data, test_data, valid_data = load_data('dX_Y')
    hidden_dim = 7
    number_of_layers = 1
    dropout_enc = 0.5
    dropout_dec = 0.5
    enc = Encoder(input_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_enc)
    dec = Decoder(output_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_dec)
    model = Seq2Seq(encoder=enc, decoder=dec, device=device).to(device)
    train_data = torch.tensor(train_data).transpose(1, 0).transpose(0, 2).float().to(device)
    test_data = torch.tensor(test_data).transpose(1, 0).transpose(0, 2).float().to(device)
    valid_data = torch.tensor(valid_data).transpose(1, 0).transpose(0, 2).float().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_data, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_data, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print('epoch: ', epoch, 'time: ', epoch_mins, 'mins', epoch_secs,'secs')
        print('train loss: ', train_loss)
        print('valid loss: ', valid_loss)

    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss = evaluate(model, test_data, criterion)
    print('test loss: ', test_loss)


if __name__ == '__main__':
    run()
