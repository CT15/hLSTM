import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_data_loader(inputs, labels, batch_size, shuffle=False):
    labels = np.array(labels)
    labels = torch.from_numpy(labels)

    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).type('torch.FloatTensor')

    data = TensorDataset(inputs, labels)
    
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=True)
