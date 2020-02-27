import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Dict

from core.common import *


def preprocess(raw_dataset: VehicleDataset, window_size: int, shift: int) -> Dict[str, Trajectories]:
    vehicle_objects = raw_dataset.vehicle_objects
    number = 0
    number_left = 0
    number_right = 0
    left_iter = []
    right_iter = []
    keep_iter = []
    total_idx = 0
    number_of_features = 3
    features = np.zeros((number_of_features, window_size))

    # New variables for feature extraction
    delta_X = np.zeros(window_size)
    X = np.zeros(window_size)
    Y = np.zeros(window_size)
    V = np.zeros(window_size)
    A = np.zeros(window_size)

    # New containers for features
    N = int(window_size / shift)
    #
    for idx, l_vehicle in enumerate(vehicle_objects):
        # TODO: Az iterátorokat tagfüggvény hozza létre
        if N * window_size > l_vehicle.size:
            continue
        # lane_change_idx, labels = lane_change_to_idx(vehicle)
        if l_vehicle.lane_change_indicator() is None:
            continue
        else:
            lane_change_idx, indicator = l_vehicle.indicator
        if lane_change_idx > 3 * window_size:
            # print(vehicle.id)
            if indicator == 1:
                number_right += 1
                right_iter.append(idx)
            if indicator == -1:
                number_left += 1
                left_iter.append(idx)
        if lane_change_idx == 0:
            keep_iter.append(idx)
            number += 1

    raw_dataset.left_iter = left_iter
    raw_dataset.right_iter = right_iter
    raw_dataset.keep_iter = keep_iter

    number_of_features = 1
    dX_left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    dX_keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    dX_right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    number_of_features = 1
    X_left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    X_keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    X_right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    number_of_features = 2
    dX_Y_left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    dX_Y_keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    dX_Y_right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    number_of_features = 2
    X_Y_left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    X_Y_keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    X_Y_right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    number_of_features = 3
    dX_V_A_left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    dX_V_A_keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    dX_V_A_right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    # left_data.shape
    # keep_data.shape
    # right_data.shape

    print(len(left_iter), len(keep_iter), len(right_iter))
    print(number_left, number, number_right)

    for left in left_iter:
        lane_change_idx, labels = vehicle_objects[left].indicator
        for k in range(N):

            index = lane_change_idx - 2 * window_size + k * shift + 1
            delta_X = (vehicle_objects[left].x[index: index + window_size]
                           - vehicle_objects[left].x[index - 1: index + window_size - 1])
            X = vehicle_objects[left].x[index: index + window_size]
            Y = vehicle_objects[left].y[index: index + window_size]
            V = vehicle_objects[left].v[index: index + window_size]
            A = vehicle_objects[left].a[index: index + window_size]

            dX_left_data[total_idx] = delta_X
            X_left_data[total_idx] = X
            dX_Y_left_data[total_idx] = np.array([delta_X, Y])
            X_Y_left_data[total_idx] = np.array([X, Y])
            dX_V_A_left_data[total_idx] = np.array([delta_X, V, A])
            total_idx += 1
    # np.savetxt("left0.csv", left_data, delimiter=",")
    total_idx = 0
    for right in right_iter:
        lane_change_idx, labels = vehicle_objects[right].indicator
        for k in range(N):

            index = lane_change_idx - 2 * window_size + k * shift + 1
            delta_X = (vehicle_objects[right].x[index: index + window_size]
                       - vehicle_objects[right].x[index - 1: index + window_size - 1])
            X = vehicle_objects[right].x[index: index + window_size]
            Y = vehicle_objects[right].y[index: index + window_size]
            V = vehicle_objects[right].v[index: index + window_size]
            A = vehicle_objects[right].a[index: index + window_size]

            dX_right_data[total_idx] = delta_X
            X_right_data[total_idx] = X
            dX_Y_right_data[total_idx] = np.array([delta_X, Y])
            X_Y_right_data[total_idx] = np.array([X, Y])
            dX_V_A_right_data[total_idx] = np.array([delta_X, V, A])
            total_idx += 1

    # np.savetxt("right0.csv", right_data, delimiter=",")
    total_idx = 0

    for keep in keep_iter:
        lane_change_idx, labels = vehicle_objects[keep].indicator
        for k in range(N):

            index = lane_change_idx - 2 * window_size + k * shift + 1
            delta_X = (vehicle_objects[keep].x[index: index + window_size]
                       - vehicle_objects[keep].x[index - 1: index + window_size - 1])
            X = vehicle_objects[keep].x[index: index + window_size]
            Y = vehicle_objects[keep].y[index: index + window_size]
            V = vehicle_objects[keep].v[index: index + window_size]
            A = vehicle_objects[keep].a[index: index + window_size]

            dX_keep_data[total_idx] = delta_X
            X_keep_data[total_idx] = X
            dX_Y_keep_data[total_idx] = np.array([delta_X, Y])
            X_Y_keep_data[total_idx] = np.array([X, Y])
            dX_V_A_keep_data[total_idx] = np.array([delta_X, V, A])
            total_idx += 1

    # reshape the arrays in order to block the trajectories by vehicles
    dX_left_data = np.reshape(dX_left_data, (len(left_iter), N, 1, window_size))
    dX_right_data = np.reshape(dX_right_data, (len(right_iter), N, 1, window_size))
    dX_keep_data = np.reshape(dX_keep_data, (len(keep_iter), N, 1, window_size))
    # reshape the arrays in order to block the trajectories by vehicles
    X_left_data = np.reshape(X_left_data, (len(left_iter), N, 1, window_size))
    X_right_data = np.reshape(X_right_data, (len(right_iter), N, 1, window_size))
    X_keep_data = np.reshape(X_keep_data, (len(keep_iter), N, 1, window_size))
    # reshape the arrays in order to block the trajectories by vehicles
    dX_Y_left_data = np.reshape(dX_Y_left_data, (len(left_iter), N, 2, window_size))
    dX_Y_right_data = np.reshape(dX_Y_right_data, (len(right_iter), N, 2, window_size))
    dX_Y_keep_data = np.reshape(dX_Y_keep_data, (len(keep_iter), N, 2, window_size))
    # reshape the arrays in order to block the trajectories by vehicles
    X_Y_left_data = np.reshape(X_Y_left_data, (len(left_iter), N, 2, window_size))
    X_Y_right_data = np.reshape(X_Y_right_data, (len(right_iter), N, 2, window_size))
    X_Y_keep_data = np.reshape(X_Y_keep_data, (len(keep_iter), N, 2, window_size))
    # reshape the arrays in order to block the trajectories by vehicles
    dX_V_A_left_data = np.reshape(dX_V_A_left_data, (len(left_iter), N, 3, window_size))
    dX_V_A_right_data = np.reshape(dX_V_A_right_data, (len(right_iter), N, 3, window_size))
    dX_V_A_keep_data = np.reshape(dX_V_A_keep_data, (len(keep_iter), N, 3, window_size))

    dX_traject = Trajectories(dX_left_data, dX_right_data, dX_keep_data, window_size, shift, featnumb=1)
    X_traject = Trajectories(X_left_data, X_right_data, X_keep_data, window_size, shift, featnumb=1)
    dX_Y_traject = Trajectories(dX_Y_left_data, dX_Y_right_data, dX_Y_keep_data, window_size, shift, featnumb=2)
    X_Y_traject = Trajectories(X_Y_left_data, X_Y_right_data, X_Y_keep_data, window_size, shift, featnumb=2)
    dX_V_A_traject = Trajectories(dX_V_A_left_data, dX_V_A_right_data, dX_V_A_keep_data, window_size, shift, featnumb=3)

    return {"dX": dX_traject,
            "X": X_traject,
            "dX_Y": dX_Y_traject,
            "X_Y": X_Y_traject,
            "dX_V_A": dX_V_A_traject}


def run():
    i_80 = "../../../full_data/i-80.csv"
    us_101 = "../../../full_data/us-101.csv"
    raw_dataset_1 = VehicleDataset(us_101)
    raw_dataset_1.create_vehicle_objects()
    # print(raw_dataset.vehicle_objects[0].lane_id, raw_dataset.vehicle_objects[1].lane_id)
    window_size = 30
    shift = 5
    dict_trajectories_1 = preprocess(raw_dataset_1, window_size, shift)
    raw_dataset_2 = VehicleDataset(i_80)
    raw_dataset_2.create_vehicle_objects()
    dict_trajectories_2 = preprocess(raw_dataset_2, window_size, shift)
    for key in dict_trajectories_1:
        traject = dict_trajectories_1[key] + dict_trajectories_2[key]
        traject.create_dataset()
        traject.save_np_dataset_labels(name=key)


if __name__ == '__main__':
    run()
