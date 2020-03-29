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

import logging
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
# print(FULLDATA_PATH)
path = None
# if not os.path.exists(path):
#     os.makedirs(path)


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
    # output = l_model(src)

    # output = [trg len, batch size, output dim]
    # todo: reverse the output tensor

    output_dim = output.shape[-1]
    # output = output.view(-1, output_dim)
    # trg = trg.view(-1, output_dim)
    loss = l_criterion(output, trg)
    # d_output = (output[1:, :, :] - output[0:-1, :, :]).view(-1, output_dim)
    # d_target = (trg[1:, :, :] - trg[0:-1, :, :]).view(-1, output_dim)
    # loss = loss + l_criterion(d_output, d_target)
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(l_model.parameters(), l_clip)

    l_optimizer.step()

    return loss.item()


def evaluate(model, valid_data, criterion, test=False, best=False, dir=None):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        src = valid_data
        trg = valid_data

        output = model(src, trg, 0)  # turn off teacher forcing
        # output = model(src)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        shape = output.shape
        output_dim = shape[-1]
        # output = output.view(-1, output_dim)
        # trg = trg.view(-1, output_dim)
        loss = criterion(output, trg)
        # d_output = (output[1:, :, :] - output[0:-1, :, :]).view(-1, output_dim)
        # d_target = (trg[1:, :, :] - trg[0:-1, :, :]).view(-1, output_dim)
        # loss = loss + criterion(d_output, d_target)

        if test:
            torch.save(trg.view(shape), os.path.join(path, dir) + '/target.pt')
            torch.save(output.view(shape), os.path.join(path, dir) + '/output.pt')
            print(output.shape)

        if best:
            torch.save(trg.view(shape), os.path.join(path, dir) + '/train_target.pt')
            torch.save(output.view(shape), os.path.join(path, dir) + '/train_output.pt')
            print(output.shape)

    return loss.item()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    elapsed_milisecs = int((elapsed_time - elapsed_mins * 60 - elapsed_secs) * 1000)
    return elapsed_mins, elapsed_secs, elapsed_milisecs


def run_lstm(N_EPOCHS=5, CLIP=1, q=0.1, hidden_dim=60, number_of_layers=4, dropout_enc=0.5,
             dropout_dec=0.5):
    # N_EPOCHS = 300
    # CLIP = 1
    #data split ratio
    # q = 0.1
    global path
    path = os.path.join(RESULTS_PATH, "seq2seq_lstm")
    if not os.path.exists(path):
        os.makedirs(path)

    best_valid_loss = float('inf')
    best_epoch_number = 0

    N, feature_dim, seq_length, train_data, test_data, valid_data = load_data('X_Yfull', q=q)
    # hidden_dim = 10
    # number_of_layers = 3
    # dropout_enc = 0.5
    # dropout_dec = 0.5
    directory = 'norm_derivative05_full__hid' + str(hidden_dim) + '_layer' + str(number_of_layers) + '_drop' + \
                str(dropout_dec).replace('.', '') + '_epoch' + str(N_EPOCHS)
    enc = EncoderLSTM(input_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_enc)
    dec = DecoderLSTM(output_dim=feature_dim, hid_dim=hidden_dim, n_layers=number_of_layers, dropout=dropout_dec)
    model = Seq2Seq(encoder=enc, decoder=dec, device=LOCAL_DEVICE).to(LOCAL_DEVICE)
    train_data = torch.tensor(train_data).transpose(1, 0).transpose(0, 2).float().to(LOCAL_DEVICE)
    test_data = torch.tensor(test_data).transpose(1, 0).transpose(0, 2).float().to(LOCAL_DEVICE)
    valid_data = torch.tensor(valid_data).transpose(1, 0).transpose(0, 2).float().to(LOCAL_DEVICE)
    # normalize the data sample wise
    # every sequence is divided by the max value of the sequence by every feature
    train_data = (train_data - train_data.min(dim=1, keepdim=True)[0]) / train_data.max(dim=1, keepdim=True)[0]
    test_data = (test_data - test_data.min(dim=1, keepdim=True)[0]) / test_data.max(dim=1, keepdim=True)[0]
    valid_data = (valid_data - valid_data.min(dim=1, keepdim=True)[0]) / valid_data.max(dim=1, keepdim=True)[0]

    optimizer = optim.Adam(model.parameters())
    # criterion = nn.MSELoss()
    criterion = CustomLoss(0.5)
    # criterion = weighted_MSEloss
    # criterion = extended_MSEloss
    print(LOCAL_DEVICE)
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
            model_params_file_name = os.path.join(path, directory) + '/model_parameters.pt'
            torch.save(model.state_dict(), model_params_file_name)
            best_epoch_number = epoch

        print('epoch: ', epoch, 'time: ', epoch_mins, 'mins', epoch_secs,'secs', epoch_milisecs, 'mili secs')
        print('train loss: ', train_loss)
        print('valid loss: ', valid_loss)

    model.load_state_dict(torch.load(model_params_file_name))

    test_loss = evaluate(model, test_data, criterion, test=True, dir=directory)
    best_train_loss = evaluate(model, train_data, criterion, best=True, dir=directory)
    print('best epoch number: ', best_epoch_number)
    print('best valid loss: ', best_valid_loss)
    print('corresponding train loss: ', best_train_loss)
    print('test loss: ', test_loss)
    logger = logging.getLogger('logfile')
    hdlr = logging.FileHandler(os.path.join(path, directory) + '/logfile.log')
    formatter = logging.Formatter('%(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info('best epoch number: ' + str(best_epoch_number))
    logger.info('best valid loss: ' + str(best_valid_loss))
    logger.info('corresponding train loss: ' + str(best_train_loss))
    logger.info('test loss: ' + str(test_loss))


def run_simple(N_EPOCHS=25000, CLIP=1, q=0.1, hidden_dim=10):
    # N_EPOCHS = 300
    # CLIP = 1
    #data split ratio
    # q = 0.1

    global path
    path = os.path.join(RESULTS_PATH, "seq2seq_simple")
    if not os.path.exists(path):
        os.makedirs(path)

    best_valid_loss = float('inf')
    best_epoch_number = 0

    N, feature_dim, seq_length, train_data, test_data, valid_data = load_data('X_Yfull', q=q)
    # hidden_dim = 10
    # number_of_layers = 3
    # dropout_enc = 0.5
    # dropout_dec = 0.5
    directory = 'norm_derivative05_full__hid' + str(hidden_dim) + '_epoch' + str(N_EPOCHS)
    enc = EncoderSimple(input_channels=feature_dim, seq_length=seq_length, context_dim=hidden_dim)
    dec = DecoderSimple(output_channels=feature_dim, seq_length=seq_length, context_dim=hidden_dim)
    model = AutoEncoder(encoder=enc, decoder=dec).to(LOCAL_DEVICE)
    train_data = torch.tensor(train_data).float().to(LOCAL_DEVICE)
    test_data = torch.tensor(test_data).float().to(LOCAL_DEVICE)
    valid_data = torch.tensor(valid_data).float().to(LOCAL_DEVICE)

    # normalize the data sample wise
    # every sequence is divided by the max value of the sequence by every feature
    train_data = (train_data - train_data.min(dim=1, keepdim=True)[0]) / train_data.max(dim=1, keepdim=True)[0]
    test_data = (test_data - test_data.min(dim=1, keepdim=True)[0]) / test_data.max(dim=1, keepdim=True)[0]
    valid_data = (valid_data - valid_data.min(dim=1, keepdim=True)[0]) / valid_data.max(dim=1, keepdim=True)[0]

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    # criterion = CustomLoss(0.5)
    # criterion = weighted_MSEloss
    # criterion = extended_MSEloss
    print(LOCAL_DEVICE)
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
            model_params_file_name = os.path.join(path, directory) + '/model_parameters.pt'
            torch.save(model.state_dict(), model_params_file_name)
            best_epoch_number = epoch

        print('epoch: ', epoch, 'time: ', epoch_mins, 'mins', epoch_secs,'secs', epoch_milisecs, 'mili secs')
        print('train loss: ', train_loss)
        print('valid loss: ', valid_loss)

    model.load_state_dict(torch.load(model_params_file_name))

    test_loss = evaluate(model, test_data, criterion, test=True, dir=directory)
    best_train_loss = evaluate(model, train_data, criterion, best=True, dir=directory)
    print('best epoch number: ', best_epoch_number)
    print('best valid loss: ', best_valid_loss)
    print('corresponding train loss: ', best_train_loss)
    print('test loss: ', test_loss)
    logger = logging.getLogger('logfile')
    hdlr = logging.FileHandler(os.path.join(path, directory) + '/logfile.log')
    formatter = logging.Formatter('%(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info('best epoch number: ' + str(best_epoch_number))
    logger.info('best valid loss: ' + str(best_valid_loss))
    logger.info('corresponding train loss: ' + str(best_train_loss))
    logger.info('test loss: ' + str(test_loss))


if __name__ == '__main__':
    # run_lstm(N_EPOCHS=60000, CLIP=1, q=0.1, hidden_dim=60, number_of_layers=4, dropout_enc=0.5,
    #          dropout_dec=0.5)
    # run_lstm(N_EPOCHS=30000, CLIP=1, q=0.1, hidden_dim=30, number_of_layers=4, dropout_enc=0.5,
    #          dropout_dec=0.5)
    # run_lstm(N_EPOCHS=25000, CLIP=1, q=0.1, hidden_dim=20, number_of_layers=4, dropout_enc=0.5,
    #          dropout_dec=0.5)
    run_lstm(N_EPOCHS=7)
