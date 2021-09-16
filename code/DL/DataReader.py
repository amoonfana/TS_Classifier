import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeries(Dataset):
    def __init__(self, file_name, transform=None):
        data = np.loadtxt(file_name, dtype=np.float32)
        self.data_len = data.shape[0]
        self.ts_len = data.shape[1] - 1
        
        X = data[:, 1:]
        #Z-score norm, may it be optmized?
        # for i in range(X.shape[0]):
        #     X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])
        
        self.X = torch.from_numpy(X).unsqueeze(1)
        
        Y = data[:, 0]
        #Some classes start from 0, yet some others starts from 1 or even bigger
        #Some classes are even -1 and 1
        Y_min = Y.min(0)
        if Y_min > 0:
            Y -= Y_min
        elif Y_min < 0:
            Y[Y<0] = 0
        
        self.Y = torch.from_numpy(Y.astype(np.int64))
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        
    def __len__(self):
        return self.data_len
