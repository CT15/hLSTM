import torch
from torch import nn
from torchcrf import CRF

from hLSTM import hLSTM

class hLSTM_CRF(nn.Module):
    def __init__(self, num_tags, input_size, hidden_size, output_size, batch_size,
                 num_layers, bidirectional, embedding, drop_prob=0.5, max_output=20, 
                 device=torch.device('cpu')):

        super().__init__()

        self.crf = CRF(num_tags=num_tags, batch_first=True)

        self.hlstm = hLSTM(input_size, hidden_size, output_size, batch_size,
                           num_layers, bidirectional, embedding, drop_prob, max_output, 
                           device, out_features=num_tags)
        
        self.batch_size = batch_size
        self.max_output = max_output
        self.num_tags = num_tags

        self.crf.to(device)
        self.to(device)


    def forward(self, inputs, mask=None):
        emissions = self.hlstm(inputs)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path


    def loss(self, inputs, labels, mask=None):
        emissions = self.hlstm(inputs)
        emissions = torch.reshape(emissions, (self.batch_size, self.max_output, self.num_tags))
        nll = -self.crf(emissions, labels, mask)
        return nll # negative log likelihood
