from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

# import math
import time
from core.utils import *
from core import common

import logging
import warnings

from core.utils import *

# from skimage import io, transform
# from torchvision import transforms, utils
warnings.filterwarnings("ignore")


SEED = 420 + 911

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class TrainingBase:
    def __init__(self, name, seed=1331):
        self.name = name
        self.model = None
        self.path = None
        self.seed = seed
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.device = common.LOCAL_DEVICE

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def load_data(self):
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        elapsed_milisecs = int((elapsed_time - elapsed_mins * 60 - elapsed_secs) * 1000)
        return elapsed_mins, elapsed_secs, elapsed_milisecs

    def set_path(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def run(self):
        pass



