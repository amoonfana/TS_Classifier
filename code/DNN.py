# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler

from DL.DataReader import TimeSeries
from Net.ROCKET import ROCKET
from Net.ConvNet1d import ConvNet1d
from Net.ResNet1d import resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152
from Net.InceptionTime import InceptionNet
from DL.Loss import LinearCE, ModifiedCE, CenterLoss, SphereFace, NormFace, ArcMarginProduct, QAMFace
# from DL.Loss import L2Regularization
from DL.Trainer import fit

dataset_info = pd.read_csv("ds_info.csv")
results = pd.DataFrame(index = dataset_info["dataset"], columns = ["acc_mean", "acc_std", "acc_best", "time_train", "time_val"], data = 0)

# Set up hyper-parameters
n_runs = 5
n_output = 128
learning_rate = 1e-3
n_epochs = 256  #256 could be better
batch_size = 128
patience = 80
log_interval = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True}

for _, row in dataset_info.iterrows():
    dataset = row["dataset"]
    n_classes = row["numClasses"]
    
    TS_Train = TimeSeries("../../UCRArchive_2018/{}/{}_TRAIN.tsv".format(dataset, dataset))
    TS_Test = TimeSeries("../../UCRArchive_2018/{}/{}_Test.tsv".format(dataset, dataset))
    
    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(TS_Train, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(TS_Test, batch_size=batch_size, shuffle=False, **kwargs)
    
    accs = np.zeros(n_runs)
    times_train = np.zeros(n_runs)
    times_val = np.zeros(n_runs)
    for i in range(n_runs):
        # Set up networks, loss functions and optimizers
        # model = ROCKET(1, TS_Train.ts_len, n_output//2) # n_output must be multiple of 2
        # model = ConvNet1d(n_output, TS_Train.ts_len)
        # model = resnet1d18(n_output)
        model = InceptionNet(1, n_output)
        
        # loss_fn = QAMFace(n_output, n_classes)
        # loss_fn = ArcMarginProduct(n_output, n_classes)
        # loss_fn = NormFace(n_output, n_classes)
        # loss_fn = SphereFace(n_output, n_classes)
        # loss_fn = CenterLoss(n_output, n_classes)
        # loss_fn = ModifiedCE(n_output, n_classes)
        loss_fn = LinearCE(n_output, n_classes)
        
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}, {'params': loss_fn.parameters(), 'lr': learning_rate}], weight_decay=0.01)
        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': learning_rate}, {'params': loss_fn.parameters(), 'lr': learning_rate}], momentum=0.8)
        
        scheduler = lr_scheduler.StepLR(optimizer, 40, gamma=0.5, last_epoch=-1)
        
        # Train the network
        accs[i], times_train[i], times_val[i] = fit(train_loader,
                                                           val_loader,
                                                           model,
                                                           loss_fn,
                                                           optimizer,
                                                           scheduler,
                                                           n_epochs,
                                                           patience,
                                                           device,
                                                           "../models/{}.pkl".format(dataset),
                                                           log_interval)
        
        torch.cuda.empty_cache()
    
    results.loc[dataset, "acc_mean"] = accs.mean()
    results.loc[dataset, "acc_std"] = accs.std()
    results.loc[dataset, "acc_best"] = accs.max()
    results.loc[dataset, "time_train"] = times_train.mean()
    results.loc[dataset, "time_val"] = times_val.mean()
    
results.to_csv("../results/DNN.csv")