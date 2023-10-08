import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import pandas as  pd
import time
import sys
from sklearn.metrics import average_precision_score
from A02_Dataset import *

class ResidualBlock(nn.Module):
    def __init__(self, N, W, D):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(N)
        self.bn2 = nn.BatchNorm1d(N)
        self.conv1 = nn.Conv1d(N, N, W, dilation=D, padding='same')
        self.conv2 = nn.Conv1d(N, N, W, dilation=D, padding='same')
    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return(out)

class NeuralNetwork(nn.Module):
    def __init__(self, NWD=[[32, 11, 1]] * 4, nChannels=32, nBlocks=4, flank_size=40):
        super().__init__()
        self.NWD = NWD
        self.nC = nChannels
        self.nB = nBlocks
        self.flank_size = flank_size

        self.conv1 = nn.Conv1d(4, self.nC, 1)
        self.conv2 = nn.Conv1d(self.nC, self.nC, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(self.NWD)):
            n,w,d = self.NWD[i]
            self.resblocks.append(ResidualBlock(n, w, d))
            if (i+1)%self.nB == 0:
                self.convs.append(nn.Conv1d(self.nC, self.nC, 1))
        self.bn1 = nn.BatchNorm1d(self.nC)
        self.conv3 = nn.Conv1d(self.nC, 3, 1)
 
    def forward(self, x):
        out = self.conv1(x)
        skip = self.conv2(out)
        for i in range(len(self.NWD)):
            n,w,d = self.NWD[i]
            out = self.resblocks[i](out)
            j = 0
            if (i+1)%self.nB == 0:
                cv = self.convs[j](out)
                skip = cv + skip
                j += 1
        skip = nn.functional.pad(skip, [-self.flank_size, -self.flank_size])
        skip = self.bn1(skip)
        out = self.conv3(skip)
        out = torch.softmax(out, dim=1)
        return(out)

def loss_fn(pred, y, epsilon=1e-10):
    s1 = y[:, 0, :]*torch.log(pred[:, 0, :] + epsilon)
    s2 = y[:, 1, :]*torch.log(pred[:, 1, :] + epsilon)
    s3 = y[:, 2, :]*torch.log(pred[:, 2, :] + epsilon)
    s = s1 + s2 + s3
    return(-torch.mean(s))

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach().item() 

    total_loss = total_loss/len(dataloader)
    return([total_loss])

def val_loop(dataloader, model, loss_fn, metrics=False):
    model.eval()

    yT1 = []
    yP1 = []
    yT2 = []
    yP2 = []
    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.detach().item()
            
            if metrics:
                wh = y.sum(axis=(1,2)) >= 1
                yt1 = y[wh,1,:].flatten()
                yp1 = pred[wh,1,:].flatten()
                yt2 = y[wh,2,:].flatten()
                yp2 = pred[wh,2,:].flatten()
                yT1 += yt1
                yP1 += yp2
                yT2 += yt2
                yP2 += yp2

    total_loss = total_loss/len(dataloader)

    if not metrics:
        return([total_loss])
    else:
        acuprc1 = AUPRC(yP1, yT1)
        acuprc2 = AUPRC(yP2, yT2)
        accuracy1, threshold1 = TopK(yP1, yT1)
        accuracy2, threshold2 = TopK(yP2, yT2)
        return([total_loss, [acuprc1, accuracy1, threshold1], [acuprc2, accuracy2, threshold2]])

def AUPRC(y_pred, y_true): 
    auprc = average_precision_score(y_true, y_pred)
    return(auprc)

def TopK(y_pred, y_true, k=1):
    idx_true = np.nonzero(y_true == 1)[0]
    y_pred_argsorted = np.argsort(y_pred)
    y_pred_sorted = np.sort(y_pred)

    th = int(k*len(idx_true))
    idx_pred = y_pred_argsorted[-th:]
    accuracy = len(np.intersect1d(idx_true, idx_pred))/min(len(idx_pred), len(idx_true))
    threshold = y_pred_sorted[-th]
    return([accuracy, threshold])

def count_parameters(model):
    L = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            pnum = parameter.numel()
            L.append([name, pnum])
    df = pd.DataFrame(L)
    df.columns = ['name', 'pnum']
    print(f'Total Trainable Parameters: {df["pnum"].sum()}')
    
#################

nFilters=32
nResidualBlocks=4
NWD1 = [[nFilters, 11, 1]] * nResidualBlocks
NWD2 = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks
NWD3 = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks + [[nFilters, 21, 10]] * nResidualBlocks
NWD4 = [[nFilters, 11, 1]] * nResidualBlocks + [[nFilters, 11, 4]] * nResidualBlocks + [[nFilters, 21, 10]] * nResidualBlocks + [[nFilters, 41, 25]] * nResidualBlocks
batch_size = 32
ouFpre = __file__.split('.py')[0]

device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device} device")

#################


def fit(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000.txt', epochs=10, NWD=None, train=True, val=False):
    #flank_size = 40
    flank_size = int(inF.split('.txt')[0].split('flank')[-1])

    dataset_train = torch.load(inF.split('.txt')[0] + '_train.pth')
    dataset_val = torch.load(inF.split('.txt')[0] + '_val.pth')
    ds_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    ds_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = NeuralNetwork(NWD, flank_size=flank_size).to(device)
    count_parameters(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    lambda1 = lambda epoch: (1 if epoch <= 5 else 0.5 ** (epoch-5))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    ########
    
    H = []
    for t in range(epochs):
        train_metrics = [None]
        val_metrics = [None]
    
        if train:
            train_metrics = train_loop(ds_train, model, loss_fn, optimizer)
        if val:
            val_metrics = val_loop(ds_val, model, loss_fn)

        h = [t+1, scheduler.get_last_lr(), train_metrics[0], val_metrics[0]]
        H.append(H)
        scheduler.step()

        print(h)
        sys.stdout.flush()
    
    H = pd.DataFrame(H)
    H.columns = ['epoch', 'lr', 'train_loss', 'val_loss']
    H.to_csv(ouFpre + '_history.txt', header=True, index=False, sep='\t')
    
    torch.save(model, inF.split('.txt')[0] + '_model.pth')


def load_weights_predict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000.txt'):
    flank_size = int(inF.split('.txt')[0].split('flank')[-1])

    dataset_train = torch.load(inF.split('.txt')[0] + '_train.pth')
    dataset_val = torch.load(inF.split('.txt')[0] + '_val.pth')
    ds_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    ds_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2)

    model = torch.load(inF.split('.txt')[0] + '_model.pth', map_location=device)
    val_metrics = val_loop(ds_val, model, loss_fn, metrics=True)
    print(val_metrics)


fit(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.txt', epochs=10, NWD=NWD4)
load_weights_predict('Homo_sapiens.GRCh38.110.bed12_seq_nt40_flank40.txt')
