import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_data_loader(inputs, labels, batch_size, shuffle=False):
    labels = np.array(labels)
    inputs = np.array(inputs)
    data = TensorDataset(torch.from_numpy(inputs).type('torch.FloatTensor'), torch.from_numpy(labels))
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=True)