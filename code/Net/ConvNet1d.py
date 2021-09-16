import torch.nn as nn
import torch.nn.functional as F

class ConvNet1d(nn.Module):
    def __init__(self, n_output, ts_len):
        super(ConvNet1d, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(32, 64, 3, padding=1), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * int(ts_len/4), 256),
                                # nn.Dropout(0.9),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                # nn.Dropout(0.9),
                                nn.PReLU(),
                                nn.Linear(128, n_output))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)