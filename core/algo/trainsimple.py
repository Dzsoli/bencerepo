from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from core.utils import *
from core.common import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

lr = 0.001
num_epochs = 3000

data = np.load('../../../full_data/dataset.npy')
labels = np.load('../../../full_data/labels.npy')

# quotient for chopping the data
q = 0.15

# train data
train_data = data[0:int((1 - 2*q) * data.shape[0])]
train_labels = labels[0:int((1 - 2*q) * data.shape[0])]

# validation data
valid_data = data[int((1 - 2*q) * data.shape[0]):int((1 - q) * data.shape[0])]
valid_labels = labels[int((1 - 2*q) * data.shape[0]):int((1 - q) * data.shape[0])]

# test data
test_data = data[int((1 - q) * data.shape[0]):]
test_labels = labels[int((1 - q) * data.shape[0]):]

train_data = torch.from_numpy(train_data).float().to(device)
print(train_data.size())
train_labels = torch.LongTensor(train_labels).to(device=device, dtype=torch.float)
valid_data = torch.from_numpy(valid_data).float().to(device)
valid_labels = torch.LongTensor(valid_labels).to(device=device, dtype=torch.float)

model = SimpleLSTM(3, 10)
model = model.to(device)
# loss = nn.MSELoss()
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.05)
train_error = []
valid_error = []
valid_acc = []


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    err = loss(out, train_labels)
    # print(out.shape, train_labels.shape)
    # err = CELoss(out, train_labels)
    # print(err)


    # if epoch % 50 == 0:
    #     print('out: {}'.format(out[10]), 'label: {}'.format(train_labels[10]), 'product: {}'.format(out[10] * train_labels[10]))
    err.backward()
    optimizer.step()
    acc, valid_err = testing(model, valid_data, valid_labels, loss=loss)
    train_error.append(err)
    valid_error.append(valid_err)
    # print('epoch: {}, train loss: {}'.format(epoch, err))
    valid_acc.append(acc)

test_data = torch.from_numpy(test_data).float().to(device)
test_labels = torch.LongTensor(test_labels).to(device=device, dtype=torch.float)
testing(model, test_data, test_labels)
plt.plot(train_error)
plt.ylabel("Training losses")
plt.show()
plt.plot(valid_error)
plt.ylabel("Validation losses")
plt.show()
plt.plot(valid_acc)
plt.ylabel("Validation accuracy")
plt.show()
