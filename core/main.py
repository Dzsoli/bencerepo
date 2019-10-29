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


def train_lstm():
    lr = 0.2
    num_epochs = 5000
    # batch_size = 512
    data = np.load('../../full_data/dataset.npy')
    labels = np.load('../../full_data/labels.npy')

    # model and optimizer
    model = SimpleLSTM(3, 7).to(models.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    lstm_training.run(l_data=data, l_labels=labels, l_model=model, l_loss_fn=loss_fn, l_optimizer=optimizer_adam)


if __name__ == '__main__':
    train_lstm()
