import numpy as np
import torch
from core.common import *
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn


def dataloader(vehicle_objects, window_size, shift):
    # print("dataset length: {}".format(vehicle_objects.__len__()))
    num_of_parameters = 3
    tensor_idx = 0
    total_size = 572
    N = int(window_size / shift)

    lane_change_tensor = np.zeros((total_size, num_of_parameters, window_size))
    lane_keeping_tensor = np.zeros((total_size, num_of_parameters, window_size))

    features = np.zeros((num_of_parameters, window_size))
    tt = np.zeros((num_of_parameters, window_size))

    left_seq = []
    right_seq = []
    keep_seq = []
    label_sequences = []
    data = []
    left = [1., 0., 0.]
    right = [0., 0., 1.]
    keep = [0., 1., 0.]
    # print(lane_change_tensor.shape)
    # print(lane_keeping_tensor.shape)
    # print(features.shape)
    # print(tt.shape)

    for vehicle in vehicle_objects:
        # print("Vehicle: {}, size: {}".format(vehicle.id, vehicle.size))
        lane_change_idx, label = lane_change_to_idx(vehicle)

        if (lane_change_idx - 1) > 2 * window_size:
            batch = []
            for k in range(N):

                features[0] = vehicle.x[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]\
                              - vehicle.x[lane_change_idx - window_size - k * shift: lane_change_idx - k * shift]

                features[1] = vehicle.v[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]
                features[2] = vehicle.a[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]

                batch.append(features)

            if label == -1:
                left_seq.append(batch)
            else:
                right_seq.append(batch)

        elif lane_change_idx == 0:
            batch = []
            for k in range(N):
                features[0] = vehicle.x[lane_change_idx + 1 + k * shift: lane_change_idx + 1 + k * shift + window_size]\
                              - vehicle.x[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[1] = vehicle.v[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[2] = vehicle.a[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]

                batch.append(features)

            keep_seq.append(batch)

    lab = []
    for i in range(N):
        lab.append(left)
    for i in range(N):
        lab.append(right)
    for i in range(N):
        lab.append(keep)

    for l, r, k in zip(left_seq, right_seq, keep_seq):
        batch = np.concatenate((l, r, k), axis=0)
        data.append(batch)
        label_sequences.append(lab)

    data = np.array(data).transpose((0, 1, 3, 2))
    label_sequences = np.array(label_sequences)
    return data, label_sequences


def lane_change_to_idx(vehicle):
    j = 0
    labels = 0
    lane_change_idx = 0

    while (j < vehicle.size - 1) & (lane_change_idx == 0):
        delta = vehicle.lane_id[j + 1] - vehicle.lane_id[j]
        if delta != 0:
            lane_change_idx = j
            labels = delta
            # print("Lane change idx: {}".format(lane_change_idx))
        j = j + 1

    return lane_change_idx, labels


def dataloader_2(dataset, window_size, shift):
    vehicle_objects = dataset.vehicle_objects
    number = 0
    number_left = 0
    number_right = 0
    left_iter = []
    right_iter = []
    keep_iter = []
    total_idx = 0
    features = np.zeros((3, window_size))
    data = np.zeros((1350, 3, window_size))
    N = int(window_size/shift)
    for idx, vehicle in enumerate(vehicle_objects):
        lane_change_idx, labels = lane_change_to_idx(vehicle)
        if lane_change_idx > 3 * window_size:
            # print(vehicle.id)
            if labels == 1:
                number_right += 1
                right_iter.append(idx)
            if labels == -1:
                number_left += 1
                left_iter.append(idx)
        if lane_change_idx == 0:
            keep_iter.append(idx)
            number += 1
    # print('numbers: ', number, number_right, number_left)
    samples = np.min([len(right_iter), len(left_iter), len(keep_iter)])
    data = np.zeros((samples * 3 * N, 3, window_size))
    for left, right, keep in zip(left_iter, right_iter, keep_iter):
        # lane change left
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[left])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[left].x[index: index + window_size]
                           - vehicle_objects[left].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[left].v[index: index + window_size])
            features[2] = (vehicle_objects[left].a[index: index + window_size])
            # print(features)
            data[total_idx] = features
            total_idx += 1
        # print("K")
        # lane change right
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[right])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[right].x[index: index + window_size]
                            - vehicle_objects[right].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[right].v[index: index + window_size])
            features[2] = (vehicle_objects[right].a[index: index + window_size])
            data[total_idx] = features
            total_idx += 1
        # lane keeping
        _, labels = lane_change_to_idx(vehicle_objects[keep])
        first_idx = 3 * window_size
        for k in range(N):
            features[0] = 0
            index = 2 * window_size + k * shift + 1 # 2 windows size volt
            features[0] = (vehicle_objects[keep].x[index: index + window_size]
                            - vehicle_objects[keep].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[keep].v[index: index + window_size])
            features[2] = (vehicle_objects[keep].a[index: index + window_size])
            data[total_idx] = features
            total_idx += 1

    # print("data shape", data.shape)
    # print(data[0:20])
    # data = np.array(data)
    data = data.transpose((0, 2, 1))
    # label creation
    leftlab = [1, 0, 0]
    keeplab = [0, 1, 0]
    rightlab = [0, 0, 1]
    lab = []
    for i in range(number_right):
        for j in range(N):
            lab.append(leftlab)
        for j in range(N):
            lab.append(rightlab)
        for j in range(N):
            lab.append(keeplab)

    label = np.array(lab)
    # print('shape: ', data.shape, label.shape)ű
    return data, label


def testing(model, data, labels, loss=nn.MSELoss()):
    # print('Testing the network...')
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():

        outputs = model(data)
        _, prediction = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # print(total)
        out_idx = torch.argmax(outputs, 1)
        lab_idx = torch.argmax(labels, 1)

        # print(outputs)
        # print(labels)
        # print(out_idx)
        # print(lab_idx)
        for k in range(len(out_idx)):
            total = total + 1
            if out_idx[k] == lab_idx[k]:
                correct = correct + 1
        err = loss(outputs, labels)
    # print('Accuracy on the test set: %d %%' % (100 * correct / total))
    # print('Test loss:{}'.format(err))

    return correct/total, err
