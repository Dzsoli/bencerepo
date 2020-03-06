from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

# import math
import time
from core.utils import *
from core import common

import warnings

from core.utils import *

# from skimage import io, transform
# from torchvision import transforms, utils
warnings.filterwarnings("ignore")


SEED = 420 + 911

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

path = os.path.join(RESULTS_PATH, "seq2seq_lstm")
if not os.path.exists(path):
    os.makedirs(path)


def load_data(features: str, q, path='../../../full_data/'):

    l_data = np.load(path + features + '_dataset.npy')
    # l_labels = np.load(path + features + '_labels.npy')

    # TODO: the window_size and shift (and the N=6) is not in this scope

    window_size = l_data.shape[3]
    features = l_data.shape[2]
    N = l_data.shape[1]
    l_data = np.reshape(l_data, (-1, features, window_size))
    # TODO: concatenate the N sequences to one
    # normalize

    # l_data = l_data / np.linalg.norm(l_data, axis=0)
    # l_data = np.reshape(l_data, (-1, N, features, window_size))

    # train data
    # [number of sequences, number of features, length of sequence]
    V = l_data.shape[0]
    train_data = np.reshape(l_data[0:int((1 - 2*q) * V)], (-1, features, window_size))
    test_data = np.reshape(l_data[int((1 - 2*q) * V):int((1 - q) * V)], (-1, features, window_size))
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

    output = l_model(src, trg, 0.5)

    # trg = [trg len, batch size]
    # output = [trg len, batch size, output dim]
    # todo: reverse the output tensor

    output_dim = output.shape[-1]
    output = output.view(-1, output_dim)
    trg = trg.view(-1, output_dim)
    loss = l_criterion(output, trg)

    loss.backward()

    # torch.nn.utils.clip_grad_norm_(l_model.parameters(), l_clip)

    l_optimizer.step()

    return loss.item()


def evaluate(model, valid_data, criterion, test=False, dir=None):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        src = valid_data
        trg = valid_data

        output = model(src, trg, 0)  # turn off teacher forcing

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        shape = output.shape
        output_dim = shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1, output_dim)
        loss = criterion(output, trg)

        if test:
            torch.save(trg.view(shape), os.path.join(path, dir) + '/target.pt')
            torch.save(output.view(shape), os.path.join(path, dir) + '/output.pt')
            print(output.shape)

    return loss.item()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    elapsed_milisecs = int((elapsed_time - elapsed_mins * 60 - elapsed_secs) * 1000)
    return elapsed_mins, elapsed_secs, elapsed_milisecs


def run(N_EPOCHS=5000, CLIP=1, q=0.1, hidden_dim=5, number_of_layers=1, dropout_enc=0.5,
        dropout_dec=0.5):
    # N_EPOCHS = 300
    # CLIP = 1
    #data split ratio
    # q = 0.1
    # path = "../../../results/seq2seq_lstm/"

    best_valid_loss = float('inf')
    best_epoch_number = 0

    N, feature_dim, seq_length, train_data, test_data, valid_data = load_data('X_Y', q=q)
    # hidden_dim = 10
    # number_of_layers = 3
    # dropout_enc = 0.5
    # dropout_dec = 0.5
    directory = 'hid' + str(hidden_dim) + '_layer' + str(number_of_layers) + '_drop' + \
                str(dropout_dec).replace('.', '') + '_epoch' + str(N_EPOCHS)
    enc = Encoder(input_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_enc)
    dec = Decoder(output_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_dec)
    model = Seq2Seq(encoder=enc, decoder=dec, device=device).to(device)
    train_data = torch.tensor(train_data).transpose(1, 0).transpose(0, 2).float().to(device)
    test_data = torch.tensor(test_data).transpose(1, 0).transpose(0, 2).float().to(device)
    valid_data = torch.tensor(valid_data).transpose(1, 0).transpose(0, 2).float().to(device)
    # normalize the data sample wise
    # every sequence is divided by the max value of the sequence by every feature
    train_data = train_data / train_data.max(dim=1, keepdim=True)[0]
    test_data = test_data / test_data.max(dim=1, keepdim=True)[0]
    valid_data = valid_data / valid_data.max(dim=1, keepdim=True)[0]

    optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_data, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_data, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs, epoch_milisecs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if not os.path.exists(os.path.join(path, directory)):
                os.makedirs(os.path.join(path, directory))
            file_name = os.path.join(path, directory) + '/model_parameters.pt'
            torch.save(model.state_dict(), file_name)
            best_epoch_number = epoch

        print('epoch: ', epoch, 'time: ', epoch_mins, 'mins', epoch_secs,'secs', epoch_milisecs, 'mili secs')
        print('train loss: ', train_loss)
        print('valid loss: ', valid_loss)

    model.load_state_dict(torch.load(file_name))

    test_loss = evaluate(model, test_data, criterion, test=True, dir=directory)
    print('best epoch number: ', best_epoch_number)
    print('best valid loss: ', best_valid_loss)
    print('test loss: ', test_loss)


if __name__ == '__main__':
    run()
