import numpy as np
import pickle
import os
from sklearn import svm
from sklearn import metrics as m
import matplotlib.pyplot as plt
import matplotlib.patches as pts


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
    mean = np.mean(l_data, axis=2).squeeze()
    std = np.std(l_data, axis=2).squeeze()
    l_data = np.array([mean, std]).T
    print(l_data[0:6])
    # normalize
    # l_data = l_data / np.linalg.norm(l_data, axis=0)

    l_data = np.reshape(l_data, (-1, N, 2))
    print(l_data[0])
    # train data
    train_data = np.reshape(l_data[0:int((1 - q) * l_data.shape[0])], (-1, 2))
    train_labels = np.reshape(l_labels[0:int((1 - q) * l_data.shape[0])], (-1, 3))
    train_labels = np.argmax(train_labels, axis=1) - 1

    # test data
    test_data = np.reshape(l_data[int((1 - q) * l_data.shape[0]):], (-1, 2))
    test_labels = np.reshape(l_labels[int((1 - q) * l_data.shape[0]):], (-1, 3))
    test_labels = np.argmax(test_labels, axis=1) - 1

    return train_data, train_labels, test_data, test_labels


def learn_gaussian_classifier3(l_train_data, l_train_labels):
    inv = np.linalg.inv
    eig = np.linalg.eig
    class_keep = []
    class_left = []
    class_right = []

    for dat, lab in zip(l_train_data, l_train_labels):
        if lab == 1:
            class_right.append(dat)
        elif lab == -1:
            class_left.append(dat)
        elif lab == 0:
            class_keep.append(dat)

    n_class_keep = len(class_keep)
    n_class_left = len(class_left)
    n_class_right = len(class_right)
    print("change left: ", n_class_left)
    print("change right: ", n_class_right)
    print("keep: ", n_class_keep)

    # priori
    p_class_keep = n_class_keep / (n_class_right + n_class_left + n_class_keep)
    p_class_left = n_class_left / (n_class_right + n_class_left + n_class_keep)
    p_class_right = n_class_right / (n_class_right + n_class_left + n_class_keep)
    print("p-keep: ", p_class_keep, "\n", "p-change-left: ", p_class_left, "\n", "p-class-right: ", p_class_right)

    # class means
    class_left = np.array(class_left).T
    class_right = np.array(class_right).T
    class_keep = np.array(class_keep).T
    mu_keep = np.array([np.mean(class_keep[0]), np.mean(class_keep[1])])
    mu_left = np.array([np.mean(class_left[0]), np.mean(class_left[1])])
    mu_right = np.array([np.mean(class_right[0]), np.mean(class_right[1])])

    # common covariance matrix
    sigma = np.cov(class_keep[0], class_keep[1])
    sigma += np.cov(class_left[0], class_left[1])
    sigma += np.cov(class_right[0], class_right[1])
    sigma *= (1 / 3.0)

    eig_values, eig_vectores = eig(sigma)

    # weight vectors
    w_keep = np.dot(inv(sigma), mu_keep)
    w_left = np.dot(inv(sigma), mu_left)
    w_right = np.dot(inv(sigma), mu_right)

    # balances
    w_0_keep = -0.5 * np.dot(np.dot(mu_keep.T, inv(sigma)), mu_keep) + np.log(p_class_keep)
    w_0_left = -0.5 * np.dot(np.dot(mu_left.T, inv(sigma)), mu_left) + np.log(p_class_left)
    w_0_right = -0.5 * np.dot(np.dot(mu_right.T, inv(sigma)), mu_right) + np.log(p_class_right)
    # balances vector
    w_0 = np.array([[w_0_keep, w_0_left, w_0_right]])
    # all weights matrix
    # | w_0_keep  ,w_keep_1  ,w_keep_2  |
    # | w_0_left  ,w_left_1  ,w_left_2  |
    # | w_0_right ,w_right_1 ,w_right_2 |

    weights = np.concatenate((w_0.T, np.array([w_keep, w_left, w_right])), axis=1)

    return weights


def gaussian_classifier3(l_test_data, l_test_labels, weights):

    n = l_test_data.shape[0]
    x = np.concatenate((np.array([np.ones(n)]), l_test_data.T), axis=0)
    # A[keep, left, right]
    A = np.dot(weights, x)
    p_i_x = np.exp(A) / np.sum(np.exp(A), axis=0)
    log_p_i_x = A - np.log(np.sum(np.exp(A), axis=0))
    p_i_x_pred = np.array([p_i_x[1, :], p_i_x[0, :], p_i_x[2, :]])
    pred_idx = np.argmax(p_i_x, axis=0) - 1

    sk_f1_score = m.f1_score(l_test_labels, pred_idx, average=None)
    print(sk_f1_score)

    return sk_f1_score


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_data("dX")

    gaussian_classifier3(test_data, test_labels, learn_gaussian_classifier3(train_data, train_labels))
