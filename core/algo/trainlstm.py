from __future__ import print_function, division

import torch

import core.common.models as models
import core.common.vehicle as vehicle
import core.common.loader as loader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

csv_name = "../si_data.csv"

dataset = vehicle.VehicleDataset(csv_name)
dataset.create_objects()

# hyperparameters.
# TODO: move to hyperparam.py
lr = 0.005
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = torch.from_numpy(data).float()
labels = torch.LongTensor(labels)
model = models.LSTM()
model = model.to(device)
# loss = nn.CrossEntropyLoss()
# loss = nn.MultiLabelSoftMarginLoss()
# loss = nn.NLLLoss()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    for i in range(87):
        print(i)
        input_batch = data[i].to(device)
        input_label = labels[i].to(device=device, dtype=torch.float)
        out = model(input_batch)
        # print(out)
        # print(input_label)
        err = loss(out, input_label)
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        print('epoch: {}, loss: {}'.format(epoch, err))