from __future__ import annotations

import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
# from core.common import *
from core import common

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

feetToMeters = lambda feet: float(feet) * 0.3048
converters_dict = {'Local_X': feetToMeters,
                   'Local_Y': feetToMeters,
                   'v_length': feetToMeters,
                   'v_Width': feetToMeters,
                   'v_Vel': feetToMeters,
                   'v_Acc': feetToMeters,
                   'Space_Headway': feetToMeters}


class VehicleDataset(Dataset):
    """NGSIM vehicle dataset"""

    def __init__(self, csv_file, root_dir=None, transform=None):
        # TODO: megmagyarázni a változó neveket és argumentumokat
        """
        asd
        """
        # TODO: csak akkor kell konverter ha nem a clean van betöltve
        self.all_data = pd.read_csv(csv_file, delimiter=',', header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.vehicle_objects = None
        self.left_iter = None
        self.right_iter = None
        self.keep_iter = None

    def __len__(self):
        """returns with all the number of frames of the dataset"""
        return len(self.vehicle_objects)

    def __getitem__(self, idx):
        """returns the idx_th vehicle"""
        return self.vehicle_objects[idx]

    def create_objects(self):
        i = 0
        vehicle_objects = []
        all_data = np.array(self.all_data)
        while len(all_data) > i:
            total_frames = int(all_data[i][2])
            until = i + total_frames
            data = all_data[i:until]
            vehicle = VehicleData(data)
            # vehicle.lane_changing()
            #TODO: labeling számítás
            vehicle_objects.append(vehicle)
            i = until
        self.vehicle_objects = vehicle_objects

    def create_vehicle_objects(self):
        vehicle_objects = []
        for _, vehicle_id in self.all_data.groupby('Vehicle_ID'):
            for _, vehicle_id_tf in vehicle_id.groupby('Total_Frames'):
                vehicle = VehicleData(np.array(vehicle_id_tf))
                vehicle_objects.append(vehicle)
        self.vehicle_objects = vehicle_objects


class Trajectories(Dataset):

    def __init__(self, left=None, right=None, keep=None, window_size=None, shift=None, featnumb=None,
                 csv_file=None, root_dir=None, transform=None, data=None, dataset=None, labels=None):

        if csv_file is not None:
            self.all_data = np.array(pd.read_csv(csv_file, delimiter=',', header=0))
        else:
            self.all_data = data
        self.left = left
        self.right = right
        self.keep = keep
        self.window_size = window_size
        self.shift = shift
        self.featnumb = featnumb
        self.root_dir = common.FULLDATA_PATH
        self.transform = transform
        self.vehicle_objects = None
        # TODO: dataset, labeling
        self.dataset = dataset
        self.labels = labels

        # TODO: CHECK!!! window_size, shift, is part of the structure

    def __len__(self):
        """returns with a trajectory sample"""
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        """returns with a trajectory sample"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'data': self.dataset[idx], 'label': self.labels[idx]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __add__(self, other: Trajectories):

        assert ((self.window_size == other.window_size) & (self.shift == other.shift))
        left = np.concatenate((self.left, other.left), axis=0)
        right = np.concatenate((self.right, other.right), axis=0)
        keep = np.concatenate((self.keep, other.keep), axis=0)
        # TODO: add labels
        return Trajectories(left, right, keep, self.window_size, self.shift)

    def __iadd__(self, other: Trajectories):
        assert ((self.window_size == other.window_size) & (self.shift == other.shift))
        self.left = np.concatenate((self.left, other.left), axis=0)
        self.right = np.concatenate((self.right, other.right), axis=0)
        self.keep = np.concatenate((self.keep, other.keep), axis=0)
        # TODO: iadd labels
        return self

    def create_dataset(self):
        left_size = self.left.shape[0]
        right_size = self.right.shape[0]
        keep_size = self.keep.shape[0]
        min_size = np.min((left_size, right_size, keep_size))
        np.random.seed(seed=420)
        conc = np.concatenate((self.left[np.random.choice(left_size, min_size, replace=False), :],
                               self.right[np.random.choice(right_size, min_size, replace=False), :],
                               self.keep[np.random.choice(keep_size, min_size, replace=False), :]),
                              axis=0)

        np.random.seed(seed=911)
        self.dataset = conc[np.random.permutation(conc.shape[0])]
        self.create_labels(min_size)

    def create_labels(self, size: int):
        l = np.array([1, 0, 0])
        k = np.array([0, 1, 0])
        r = np.array([0, 0, 1])

        N = int(self.window_size / self.shift)
        size = N * size
        labels = np.zeros((size * 3, 3))
        for i in range(0, size):
            labels[i] = l
            labels[i + size] = r
            labels[i + 2 * size] = k

        labels = np.reshape(labels, (-1, N, 3))
        np.random.seed(seed=911)
        self.labels = labels[np.random.permutation(labels.shape[0])]

    def write_csv(self):
        # TODO
        path = self.root_dir
        np.savetxt(path + "left.csv", self.left, delimiter=",")
        np.savetxt(path + "right.csv", self.right, delimiter=",")
        np.savetxt(path + "keep.csv", self.keep, delimiter=",")

    def save_np_array(self):
        path = self.root_dir
        np.save(path + 'left.npy', self.left)
        np.save(path + 'right.npy', self.right)
        np.save(path + 'keep.npy', self.keep)

    def save_np_dataset_labels(self, name, mode=None):
        path = self.root_dir
        if mode is not None:
            name = name + mode
        print(path + name)
        np.save(path + "/" + name + '_dataset.npy', self.dataset)
        np.save(path + "/" + name + '_labels.npy', self.labels)


class VehicleData:
    """class for Vehicles. Instantiate in VehicleDataset.__getitem__"""
    def __init__(self, data):
        # car ID
        self.id = int(data[0, 0])
        # frame ID
        self.frames = data[:, 1]
        # total frame number
        self.size = int(data[0, 2])
        # global time
        self.t = data[:, 3]
        # lateral x coordinate
        self.x = data[:, 4]
        # Longitudinal y coordinate
        self.y = data[:, 5]
        # Dimensions of the car: Length, Width
        self.dims = data[0, 8:10]
        # Type, 1-motor, 2-car, 3-truck
        self.type = int(data[0, 10])
        # Instantenous velocity
        self.v = data[:, 11]
        # Instantenous acceleration
        self.a = data[:, 12]
        # lane ID: 1 is the FARTHEST LEFT. 5 is the FARTHEST RIGHT.
        # 6 is Auxiliary lane for off and on ramp
        # 7 is on ramp
        # 8 is off ramp
        self.lane_id = data[:, 13]
        # [None] if no lane change; [+/-1, frame] if there is a lane change in the specific frame
        # [0, frame_id] or [-1, frame_id] or [1, frame_id]
        self.change_lane = None
        # mean, variance, changes or not?, frame id
        self.labels = None

        self.indicator = None

    def __getitem__(self, frame_number):
        item = []
        # returns a numpy array with features corresponding to a specific frame number. The first frame is the zeroth.
        return item

    def set_change_lane(self, l_change):
        self.change_lane = l_change

    def lane_changing(self):
        l_change = []
        total_frames = self.size

        for i in range(int(total_frames) - 1):
            if (self.lane_id[i + 1] - self.lane_id[i]) != 0:
                l_change.append([self.lane_id[i + 1] - self.lane_id[i],
                                 self.frames[i + 1]])
            else:
                l_change.append([0, self.frames[i + 1]])
        l_change = np.array(l_change)
        self.set_change_lane(l_change)

    def lane_change_indicator(self):
        j = 0
        indicator = 0
        index = 0

        while (j < self.size -1) & (index == 0):
            difference = self.lane_id[j + 1] - self.lane_id[j]
            if difference != 0:
                index = j
                indicator = difference
            j = j + 1
        self.indicator = [index, indicator]
        return self.indicator


class ToDevice(object):
    def __call__(self, sample):
        # print(sample['data'].shape)
        sample = {'data': torch.from_numpy(sample['data'].transpose((1, 0))).float().to(device),
                  'label': torch.from_numpy(sample['label']).to(device=device, dtype=torch.float)}
        return sample


if __name__ == '__main__':

    csv_file = "../../../full_data/full_data.csv"
    sorted_xlsx = "../../../full_data/full_data_sorted.xlsx"
    sorted_csv = "../../../full_data/full_data_sorted.csv"

    x = pd.read_csv(csv_file, delimiter=',', header=0)
    x = x.sort_values(by=['Vehicle_ID', 'Frame_ID'])
    x.to_csv(sorted_csv)
    # x.to_excel(sorted_csv)
    # y = pd.read_excel(sorted_csv, nrows=100)

