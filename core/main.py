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

from core import *
import warnings
warnings.filterwarnings("ignore")


def train_lstm(l_train_loader, l_test_loader, l_lr, l_num_epochs, l_neuron, l_layers):

    # model and optimizer
    model = SimpleLSTM(3, l_neuron, l_layers).to(models.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=l_lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=l_lr)

    lstm_training.run_lstm(l_train_loader=l_train_loader, l_test_loader=l_test_loader, l_model=model, l_loss_fn=loss_fn,
                           l_num_epochs=l_num_epochs, l_optimizer=optimizer_adam, l_lr=l_lr)


if __name__ == '__main__':

    # path = "../../../results"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # train_loader, test_loader = lstm_training.load_data(path='../../full_data/')
    # lr_list = [0.01, 0.02, 0.05, 0.1]
    # for rate in lr_list:
    #     epoch = 1500
    #     train_lstm(train_loader, test_loader, rate, epoch, l_neuron=14, l_layers=1)

    train_multi_svc(path='../../full_data/')
