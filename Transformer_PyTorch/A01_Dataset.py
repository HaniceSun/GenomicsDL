### Basic Transfomer Encoder only model for classification. Trying Transformer to replace conv1d in splice site prediction.
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import re

def PreProcess(inF):
    df = pd.read_table(inF, header=0, sep='\t')
    reviews, labels = [], []
    unique_tokens = set()
    for i in range(df.shape[0]):
      review = [x.lower() for x in re.findall(r"\w+", df.iloc[i]["text"])]
      if len(review) >= MAX_REVIEW_LEN:
          review = review[:MAX_REVIEW_LEN]
      else:
        for _ in range(MAX_REVIEW_LEN - len(review)):
          review.append("<pad>")
    
      reviews.append(review)
      unique_tokens.update(review)
    
      if df.iloc[i]["label"] == 'positive':
        labels.append(1)
      else:
        labels.append(0)
    
    unique_tokens = list(unique_tokens)

    for i in range(len(reviews)):
        reviews[i] = [unique_tokens.index(x) for x in reviews[i]]

    return(reviews, labels, unique_tokens)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        X = torch.tensor(X, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)
        return(X, y)

inF = 'IMDB.txt'
MAX_REVIEW_LEN = 20
reviews, labels, unique_tokens = PreProcess(inF)
 
if __name__ == '__main__':
   
    dataset = CustomDataset(reviews, labels)
    ds_train, ds_val, ds_test = random_split(dataset, [0.7, 0.2, 0.1])
    
    torch.save(ds_train, inF.split('.txt')[0] + '_train.pth')
    torch.save(ds_val, inF.split('.txt')[0] + '_val.pth')
    torch.save(ds_test, inF.split('.txt')[0] + '_test.pth')

