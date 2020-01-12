import torch
import torch.nn as nn

import utils

class hLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size,
                num_layers, bidirectional, embedding, drop_prob=0.5,
                max_output=20, device=torch.device('cpu')):
        super(hLSTM, self).__init__()

        self.device = device

        self.embedding = embedding
        self.batch_size = batch_size
        self.max_output = max_output

        self.lstm1 = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=True,
                             batch_first=True, # input and output tensors => (batch, seq, feature)
                             dropout=0, # no dropout
                             bidirectional=bidirectional)
        
        self.second_layer_input_size = (2 if bidirectional else 1) * hidden_size
        self.lstm2 = nn.LSTM(input_size=self.second_layer_input_size,
                             hidden_size=output_size,
                             num_layers=num_layers,
                             bias=True,
                             batch_first=True,
                             dropout=0,
                             bidirectional=bidirectional)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=output_size * (2 if bidirectional else 1), out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.to(device)


    def forward(self, inputs): # lstm hidden is default to zero
        # inputs.shape == (number of thread in a batch, max number of post in a thread, max no of words in a post)
        if inputs.size() != (self.batch_size, self.max_output, inputs.shape[2]):
            raise Exception(f'Expected input size of ({self.batch_size}, {self.max_output}, {inputs.shape[2]}) ' + 
                            f'but found {inputs.shape}.')
        
        # rearrange input => group all first posts together, second posts together, etc.
        inputs = torch.reshape(inputs, (self.max_output, self.batch_size, inputs.shape[2])).long()
        # inputs.shape == (number of posts, batch size, no of words in a post)

        embeds = self.embedding(inputs) # embeds.shape == (self.max_output, self.batch_size, seq length, word embedding dimension)

        second_layer_input = torch.empty((self.batch_size, self.max_output, self.second_layer_input_size),
                                         dtype=torch.float, device=self.device)

        for i in range(self.max_output):
            lstm1_out, hidden = self.lstm1(embeds[i]) # lstm1_out.shape == (self.batch_size, seq length, self.hidden_size)
            second_layer_input[:, i, :] = lstm1_out[:, -1, :]

        lstm2_out, hidden = self.lstm2(second_layer_input)

        lstm2_out = lstm2_out.contiguous().view(-1, lstm2_out.size()[2])

        out = self.dropout(lstm2_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out
