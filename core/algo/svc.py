from __future__ import print_function, division
import numpy as np
import pickle
import os
from sklearn import svm
from sklearn import metrics as m
import matplotlib.pyplot as plt
import matplotlib.patches as pts
import scipy
# from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV
import pandas as pd

from core import *

import warnings
warnings.filterwarnings("ignore")


def calculate_mean_std(l_data):
    newdata = np.zeros((l_data.shape[0], 2))
    size = l_data.shape[1]
    for i, _ in enumerate(newdata):
        newdata[i, 0] = (l_data[i, 0] - l_data[i, -1]) / size
        newdata[i, 1] = np.std(l_data[i, :])

    newdata = newdata / np.linalg.norm(newdata, axis=0)
    print(newdata.shape)
    return newdata


def split_data(path, features: str):
    l_data = np.load(path + features + '_dataset.npy')
    l_labels = np.load(path + features + '_labels.npy')

    q = 0.2
    np.random.seed(seed=1)
    # l_data = l_data[np.random.choice(l_data.shape[0], 500, replace=False), :]
    # np.random.seed(seed=1)
    # l_labels = l_labels[np.random.choice(l_labels.shape[0], 500, replace=False), :]

    # TODO: the window_size and shift (and the N=6) is not in this scope
    l_data = np.reshape(l_data, (-1, 30))
    l_labels = np.reshape(l_labels, (-1, 3))

    # normalize
    # l_data = l_data / np.linalg.norm(l_data, axis=0)
    # l_data = np.reshape(l_data, (-1, 6, 3, 30))

    # train data
    train_data = l_data[0:int((1 - q) * l_data.shape[0])]  # , 0::5]
    # choose the delta X feature only
    # train_data = calculate_mean_std(train_data)
    train_labels = l_labels[0:int((1 - q) * l_data.shape[0])]
    # transform to scalars (-1, 0, 1)
    train_labels = np.argmax(train_labels, axis=1) - 1
    train_labels = train_labels
    # print(train_labels)
    # print(train_labels.shape)

    # test data
    test_data = l_data[int((1 - q) * l_data.shape[0]):]  # , 0::5]
    # choose the delta X feature only
    # test_data = calculate_mean_std(test_data)
    test_labels = l_labels[int((1 - q) * l_data.shape[0]):]
    # transform to scalars (-1, 0, 1)
    test_labels = np.argmax(test_labels, axis=1) - 1
    return train_data, train_labels, test_data, test_labels


def train_multi_svc(path='../../../full_data/'):
    train_data, train_labels, test_data, test_labels = split_data(path, "dX")

    # size = len(vehicle_objects)
    # train_x, train_label = VehicleData.get_data(vehicle_objects[0:(size // 5) * 4:8])
    # test_x, test_label = VehicleData.get_data(vehicle_objects[(size // 5) * 4:size:8])
    # train_x = train_x - np.mean(train_x, axis=0)
    # train_x = train_x / np.std(train_x, axis=0)
    # test_x = test_x - np.mean(test_x, axis=0)
    # test_x = test_x / np.std(test_x, axis=0)
    # train_x = np.abs(train_x)
    # train_label = np.abs(train_label)
    # test_label = np.abs(test_label)
    # test_x = np.abs(test_x)
    classifier = []
    numb = 1e-1
    for j in range(1, int(1 / numb)):
        C = 1.0 - j * numb
        # C=C
        nu = 0.02 + j * numb * 0.02
        clf = svm.SVC(C=C, kernel='rbf', gamma='scale', decision_function_shape='ovo')
        # clf = svm.NuSVC(nu=nu, kernel='rbf', gamma='scale', decision_function_shape='ovo')
        print('instantiation is done')
        try:
            clf.fit(train_data, train_labels)
            print('fit is done')
            predicted_label = clf.predict(test_data)
            print('predict is done')
            true_pos = 0
            false_pos = 0
            false_neg = 0
            p = 0.3
            good = 0
            bad = 0

            if len(test_labels) == len(predicted_label):
                print("dimensions equal")
                for j in range(len(predicted_label)):
                    if predicted_label[j] == test_labels[j]:
                        good = good + 1
                    else:
                        bad = bad + 1
                    if predicted_label[j] == 1:
                        if test_labels[j] == 1:
                            true_pos += 1
                        else:
                            false_pos += 1
                    elif predicted_label[j] == 0:
                        if test_labels[j] == 1:
                            false_neg += 1
            # f_1_score = f_measure(true_pos, false_pos, false_neg, 1.)
            # f_2_score = f_measure(true_pos, false_pos, false_neg, 2.)
            # f_05_score = f_measure(true_pos, false_pos, false_neg, 0.5)
            recall = true_pos / (true_pos + false_neg)
            precision = true_pos / (true_pos + false_pos)
            sk_f1_score = m.f1_score(test_labels, predicted_label, average=None)
            print("RECALL: ", recall, "PRECISION: ", precision, "ACCURACY: ", good / (good + bad), "SKLEARN_F_1: ", sk_f1_score)
            print('C= ', C, 'TP= ', true_pos, 'FP= ', false_pos, 'FN= ', false_neg)
            classifier.append(
                [clf, C, true_pos, false_neg, false_pos, recall, precision, sk_f1_score])
        except:
            classifier.append(None)
            print('fit infeasible')
            continue


def grid_search(path='../../../full_data/'):
    C = np.power(10,np.arange(-3, 3, 0.5)).tolist()
    gamma = np.arange(1e-3, 1, 0.05).tolist()
    gamma.append('scale')
    # grid_1 = {'C': loguniform(1e-3, 1e3), 'gamma': loguniform(1e-4, 1e-1), 'kernel': ['rbf']}
    grid_2 = {'C': C, 'gamma': gamma, 'kernel': ['rbf'], 'decision_function_shape': ['ovr', 'ovo']}
    svc = svm.SVC()
    clf = GridSearchCV(svc, grid_2, n_jobs=7)

    train_data, train_labels, test_data, test_labels = split_data(path, "dX")
    clf.fit(train_data, train_labels)

    sorted(clf.cv_results_.keys())
    results = pd.DataFrame(clf.cv_results_)
    print(clf.best_estimator_)
    results.to_csv('../../../svm_results/grid2_search.csv')


def best_testing(path='../../../full_data/'):
    train_data, train_labels, test_data, test_labels = split_data(path, "dX")
    C = 3.1622776601683795
    degree = 3
    gamma = 'scale'
    kernel = 'rbf'
    svc = svm.SVC(C, kernel, degree, gamma)
    svc.fit(train_data, train_labels)
    predicted_label = svc.predict(test_data)
    true_pos = 0
    false_pos = 0
    false_neg = 0
    p = 0.3
    good = 0
    bad = 0

    if len(test_labels) == len(predicted_label):
        print("dimensions equal")
        for j in range(len(predicted_label)):
            if predicted_label[j] == test_labels[j]:
                good = good + 1
            else:
                bad = bad + 1
            if predicted_label[j] == 1:
                if test_labels[j] == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            elif predicted_label[j] != 1:
                if test_labels[j] == 1:
                    false_neg += 1
    # f_1_score = f_measure(true_pos, false_pos, false_neg, 1.)
    # f_2_score = f_measure(true_pos, false_pos, false_neg, 2.)
    # f_05_score = f_measure(true_pos, false_pos, false_neg, 0.5)
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    sk_f1_score = m.f1_score(test_labels, predicted_label, average=None)
    print("RECALL: ", recall, "PRECISION: ", precision, "F1: ", 2*recall*precision/(recall + precision), "ACCURACY: ", good / (good + bad), "SKLEARN_F_1: ", sk_f1_score)
    print('C= ', C, 'TP= ', true_pos, 'FP= ', false_pos, 'FN= ', false_neg)


if __name__ == "__main__":
    # train_multi_svc()
    # grid_search()
    best_testing()
