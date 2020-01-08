import torch
import torch.nn as nn

import utils

class hLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size,
                num_layers, bidirectional, embedding, drop_prob=0.5,
                max_output=20):
        self.embedding = embedding
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=True,
                             batch_first=True, # input and output tensors => (batch, seq, feature)
                             dropout=0, # no dropout
                             bidirectional=bidirectional)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=output_size,
                             num_layers=num_layers,
                             bias=True,
                             batch_first=True,
                             dropout=0,
                             bidirectional=bidirectional)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=output_size, out_features=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, ):




