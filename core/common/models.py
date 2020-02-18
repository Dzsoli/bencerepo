from __future__ import print_function, division

import torch
from torch._C import device

from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True,  # ez jelenti azt hogy az első dimenzióban a batch méret van
        )
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(self.hidden_size, 3)

    def forward(self, x):
        out, (h_n, h_c) = self.rnn(x, None)
        inp = out[:, -1, :]  # Return output at last time-step
        soft = self.linear(inp)
        out = self.softmax(soft)
        return out


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size=1, output_dim=3,
                 num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, hidden = self.lstm(inputs.view(len(inputs), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = lstm_out[-1].view(self.batch_size, -1)
        return y_pred.view(-1)


class LSTM2(nn.Module):
    def __init__(self, input_dims, sequence_length, cell_size, output_features=3):
        super(LSTM2, self).__init__()
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.cell_size = cell_size
        self.lstm = nn.LSTMCell(input_dims, cell_size)
        self.mlp = nn.Sequential(
            nn.Linear(cell_size, cell_size),
            nn.ReLU(),
            nn.Linear(cell_size, cell_size)
        )
        self.to_output = nn.Linear(cell_size, output_features)

    def forward(self, input):

        h_t, c_t = self.init_hidden(input.size(0))

        outputs = torch.zeroes
        print(input.size())
        print(input[0])
        print(input[1])

        for input_seq in input:
            for frame in input_seq:
                h_t, c_t = self.lstm(frame, (h_t, c_t))
                h_t = self.mlp(h_t)
                # outputs.append(self.to_output(h_t))

        return self.to_output(h_t)  # torch.cat(outputs, dim=1)

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        # embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            # nn.InstanceNorm1d(16),
            nn.PReLU(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=7, stride=1, padding=3),
            # nn.InstanceNorm1d(64),
            nn.PReLU(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=1, padding=4),
            # nn.InstanceNorm1d(128),
            nn.PReLU(128),
            nn.AvgPool1d(kernel_size=3)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=265, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1280, 560)  # output 3 classes: ...
        self.fc2 = nn.Linear(560, 3)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.drop_out(x)
        output = self.fc1(x.view(x.size()[0], -1))
        output = self.fc2(output)
        output = self.soft(output)
        return output
