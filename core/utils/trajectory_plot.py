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
    plt.gca().set_aspect("equal")
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
    print()
    local_path = RESULTS_PATH + '\seq2seq_lstm\\full__hid10_layer3_drop05_epoch20000'

    predictions = np.array(torch.load(local_path + '/output.pt', map_location='cpu'))
    targets = np.array(torch.load(local_path + '/target.pt', map_location='cpu'))
    predictions = np.transpose(predictions, (1, 0, 2))
    targets = np.transpose(targets, (1, 0, 2))
    i = 0
    for trg, pred in zip(targets, predictions):
        if i > -1:
            plot_target_pred(trg, pred)
        i += 1
    print(predictions.shape, targets.shape)


if __name__ == '__main__':
    run()
