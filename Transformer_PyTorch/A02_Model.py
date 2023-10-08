from A01_Dataset import *
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
from sklearn.metrics import average_precision_score
import os
import sys

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
  def __init__(self, num_vocab, embedding_dim, hidden_size, nheads, n_layers, max_len, num_labels, dropout):
    super().__init__()
    self.embedding = nn.Embedding(num_vocab, embedding_dim)
    self.pe = PositionalEncoding(embedding_dim, max_len = max_len)
    enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
    self.dense = nn.Linear(embedding_dim*max_len, num_labels)
    self.log_softmax = nn.LogSoftmax()

  def forward(self, x):
    x = self.embedding(x).permute(1, 0, 2)
    x = self.pe(x)
    x = self.encoder(x)
    x = x.reshape(x.shape[1], -1)
    x = self.dense(x)
    return x

def count_parameters(model):
    L = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            pnum = parameter.numel()
            L.append([name, pnum])
    df = pd.DataFrame(L)
    df.columns = ['name', 'pnum']
    print(f'Total Trainable Parameters: {df["pnum"].sum()}')

device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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

def val_loop(dataloader, model, loss_fn, optimizer):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.detach().item()

    total_loss = total_loss/len(dataloader)
    return([total_loss])

def fit(inF='IMDB.txt', epochs=10, batch_size=64, train=True, val=False):


    dataset_train = torch.load(inF.split('.txt')[0] + '_train.pth')
    dataset_val = torch.load(inF.split('.txt')[0] + '_val.pth')
    ds_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    ds_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    model = TransformerNet(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, MAX_REVIEW_LEN, NUM_LABELS, DROPOUT).to(device)
    count_parameters(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    lambda1 = lambda epoch: (1 if epoch <= 5 else 0.5 ** (epoch-5))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    H = []
    for t in range(epochs):
        train_metrics = [None]
        val_metrics = [None]

        if train:
            train_metrics = train_loop(ds_train, model, loss_fn, optimizer)
        if val:
            val_metrics = val_loop(ds_val, model, loss_fn)

        h = [t+1, scheduler.get_last_lr()[0], train_metrics[0], val_metrics[0]]
        H.append(h)
        scheduler.step()

        print(h)
        sys.stdout.flush()

    H = pd.DataFrame(H)
    H.columns = ['epoch', 'lr', 'train_loss', 'val_loss']
    H.to_csv(inF.split('.txt')[0] + '_history.txt', header=True, index=False, sep='\t')

    torch.save(model, inF.split('.txt')[0] + '_model.pth')

VOCAB_SIZE = len(unique_tokens)
HIDDEN_SIZE = 16
EMBEDDING_DIM = 30
NUM_HEADS = 3
NUM_LAYERS = 3
NUM_LABELS = 2
DROPOUT = .5
LEARNING_RATE = 1e-3

fit()
