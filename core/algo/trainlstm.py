from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from core.utils import *
from core.common import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

data = torch.from_numpy(np.load('../../../full_data/dataset.npy')).float()
labels = torch.LongTensor(np.load('../../../full_data/labels.npy'))

lr = 0.005
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.SimpleLSTM(3, 10)
model = model.to(device)
# loss = nn.CrossEntropyLoss()
# loss = nn.MultiLabelSoftMarginLoss()
# loss = nn.NLLLoss()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()

    input_batch = data.to(device)
    input_label = labels.to(device=device, dtype=torch.float)
    out = model(input_batch)
    # print(out)
    # print(input_label)
    err = loss(out, input_label)
    optimizer.zero_grad()
    err.backward()
    optimizer.step()

    print('epoch: {}, loss: {}'.format(epoch, err))
