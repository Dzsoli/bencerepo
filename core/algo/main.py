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

import warnings

from core.utils import *
from core.algo import *

# from skimage import io, transform
# from torchvision import transforms, utils
warnings.filterwarnings("ignore")


dropout = [0, 0.25, 0.5, 0.75]
# teacher_force = [0.25, 0.5, 0.75]
hidden = [10, 20, 30, 40, 50, 60]
layers = [1, 2, 3, 4]

for lay in layers:
    for hid in hidden:
        for drop in dropout:
            seq2seq.run(hidden_dim=hid,
                        number_of_layers=lay,
                        dropout_enc=drop, dropout_dec=drop)


