import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ROCKET(nn.Module):
    """RandOm Convolutional KErnel Transform
    
    ROCKET is a GPU Pytorch implementation of the ROCKET functions generate_kernels
    and apply_kernels that can be used  with univariate and multivariate time series.
    """
    
    def __init__(self, c_in, seq_len, n_kernels=1000, kss=[7, 9, 11]):
        '''
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS,
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        '''
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss
        
        # for p in self.parameters():
        #     p.requires_grad=False

    def forward(self, x):
        _output = []
        for i in range(self.n_kernels):
            out = self.convs[i](x)
            _max = out.max(dim=-1)[0]
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            _output.append(_max)
            _output.append(_ppv)
        return torch.cat(_output, dim=1)
    
# if __name__ == "__main__":
#     bs = 16
#     c_in = 1  # aka channels, features, variables, dimensions
#     c_out = 2
#     seq_len = 15
#     xb = torch.randn(bs, c_in, seq_len).cuda()
    
#     m = ROCKET(c_in, seq_len, n_kernels=1000, kss=[7, 9, 11]).cuda() # 1_000 for testing with a cpu. Default is 10k with a gpu!
    
#     dataset = "FiftyWords"
#     TS_Train = TimeSeries("../../UCRArchive_2018/{}/{}_TRAIN.tsv".format(dataset, dataset))
#     a = TS_Train.X
#     feature = m(a.cuda())
#     print(feature)