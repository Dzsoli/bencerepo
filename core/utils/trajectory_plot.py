import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.common import *


def plot_target_pred(target, prediction):
    x_target = [t[0] for t in target]
    y_target = [t[1] for t in target]

    x_pred = [t[0] for t in prediction]
    y_pred = [t[1] for t in prediction]

    fig = plt.figure()
    plt.plot(y_target, x_target)
    plt.plot(y_pred, x_pred)
    plt.show()
    del fig


# def load_files():
#     path = RESULTS_PATH
#
#
# def prepare(targ, preg):
#     return target, prediction


def run():
    print(RESULTS_PATH)
    predictions = np.array(torch.load(RESULTS_PATH + '\seq2seq_lstm\hid5_layer1_drop05_epoch5000' + '/output.pt', map_location='cpu'))
    targets = np.array(torch.load(RESULTS_PATH + '\seq2seq_lstm\hid5_layer1_drop05_epoch5000' + '/target.pt', map_location='cpu'))
    predictions = np.transpose(predictions, (1, 0, 2))
    targets = np.transpose(targets, (1, 0, 2))
    for trg, pred in zip(targets, predictions):
        plot_target_pred(trg, pred)
    print(predictions.shape, targets.shape)


if __name__ == '__main__':
    run()
