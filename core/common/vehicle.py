import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from .loader import *


class VehicleDataset(Dataset):
    """NGSIM vehicle dataset"""

    def __init__(self, csv_file, root_dir=None, transform=None):
        # TODO: megmagyarázni a változó neveket és argumentumokat
        """
        asd
        """
        self.all_data = np.array(pd.read_csv(csv_file, delimiter=',', header=None))
        self.root_dir = root_dir
        self.transform = transform
        self.vehicle_objects = None

    def __len__(self):
        """returns with all the number of frames of the dataset"""
        return len(self.vehicle_objects)

    def __getitem__(self, idx):
        """returns the idx_th vehicle"""
        return self.vehicle_objects[idx]

    def create_objects(self):
        i = 0
        vehicle_objects = []
        while len(self.all_data) > i:
            total_frames = int(self.all_data[i][2])
            until = i + total_frames
            data = self.all_data[i:until]
            vehicle = VehicleData(data)
            # vehicle.lane_changing()
            #TODO: labeling számítás
            vehicle_objects.append(vehicle)
            i = until
        self.vehicle_objects = vehicle_objects


class Trajectories(Dataset):

    def __init__(self, csv_file=None, root_dir=None, transform=None, data=None):

        if csv_file is not None:
            self.all_data = np.array(pd.read_csv(csv_file, delimiter=',', header=0))
        else:
            self.all_data = data
        self.root_dir = root_dir
        self.transform = transform
        self.vehicle_objects = None

    def __len__(self):
        """returns with a trajectory sample"""
        return len(self.vehicle_objects)

    def __getitem__(self, idx):
        """returns with a trajectory sample"""
        return self.all_data[idx]


class VehicleData:

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

    def __getitem__(self, frame_number):
        item = []
        # returns a numpy array vector with features corresponding a specific frame number. The first frame is the zero.
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


if __name__ == 'main':

    csv_name = "../si_data.csv"
    dataset = VehicleDataset(csv_name)
    dataset.create_objects()

    dataset_keep = Trajectories(dataloader(dataset))
    dataset_left = Trajectories(dataloader(dataset))
    dataset_right = Trajectories(dataloader(dataset))
