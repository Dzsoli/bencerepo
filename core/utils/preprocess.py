import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

from core.common import *


def preprocess(raw_dataset: VehicleDataset, window_size: int, shift: int) -> Trajectories:
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

    # 3 * N-ben nem kell a 3 szorzó
    left_data = np.zeros((len(left_iter) * N, number_of_features, window_size))
    keep_data = np.zeros((len(keep_iter) * N, number_of_features, window_size))
    right_data = np.zeros((len(right_iter) * N, number_of_features, window_size))

    left_data.shape
    keep_data.shape
    right_data.shape

    print(len(left_iter), len(keep_iter), len(right_iter))
    print(number_left, number, number_right)

    for left in left_iter:
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[left])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[left].x[index: index + window_size]
                           - vehicle_objects[left].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[left].v[index: index + window_size])
            features[2] = (vehicle_objects[left].a[index: index + window_size])
            # print(features)
            left_data[total_idx] = features
            total_idx += 1
    # TODO: squeeze, vagy ciklus, és/vagy numpy array
    # np.savetxt("left0.csv", left_data, delimiter=",")
    total_idx = 0
    for right in right_iter:
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[right])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[right].x[index: index + window_size]
                           - vehicle_objects[right].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[right].v[index: index + window_size])
            features[2] = (vehicle_objects[right].a[index: index + window_size])
            # print(features)
            right_data[total_idx] = features
            total_idx += 1

    # np.savetxt("right0.csv", right_data, delimiter=",")
    total_idx = 0

    for keep in keep_iter:
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[keep])
        for k in range(N):
            features[0] = 0
            index = k * shift + 1
            features[0] = (vehicle_objects[keep].x[index: index + window_size]
                           - vehicle_objects[keep].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[keep].v[index: index + window_size])
            features[2] = (vehicle_objects[keep].a[index: index + window_size])
            # print(features)
            keep_data[total_idx] = features
            total_idx += 1

    # reshape the arrays in order to block the trajectories by vehicles
    left_data = np.reshape(left_data, (len(left_iter), N, 3, window_size))
    right_data = np.reshape(right_data, (len(right_iter), N, 3, window_size))
    keep_data = np.reshape(keep_data, (len(keep_iter), N, 3, window_size))

    traject = Trajectories(left_data, right_data, keep_data, window_size, shift)
    return traject


def run():
    i_80 = "../../../full_data/i-80.csv"
    us_101 = "../../../full_data/us-101.csv"
    raw_dataset = VehicleDataset(us_101)
    raw_dataset.create_vehicle_objects()
    # print(raw_dataset.vehicle_objects[0].lane_id, raw_dataset.vehicle_objects[1].lane_id)
    window_size = 30
    shift = 5
    trajectories = preprocess(raw_dataset, window_size, shift)
    raw_dataset = VehicleDataset(i_80)
    raw_dataset.create_vehicle_objects()
    trajectories += preprocess(raw_dataset, window_size, shift)
    trajectories.create_dataset()
    trajectories.save_np_dataset_labels()


if __name__ == '__main__':
    run()
