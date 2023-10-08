import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class CustomDataset(Dataset):
    def __init__(self, inF, CH=[]):
        self.X = []
        self.y = []
        with open(inF) as inFile:
            head = inFile.readline()
            for line in inFile:
                line = line.strip()
                fields = line.split('\t')
                ch = fields[1]
                if CH:
                    if ch in CH:
                        self.X.append(fields[-2])
                        self.y.append(fields[-1])
                else:
                    self.X.append(fields[-2])
                    self.y.append(fields[-1])
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        X = self.OneHot(list(X))
        y = self.OneHot(list(y), alphabet=['0', '1', '2'])
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return(X, y)

    def OneHot(self, L, alphabet=['A', 'C', 'G', 'T']):
        cat = list(np.array(sorted(alphabet)).reshape(1, -1))
        oe = OneHotEncoder(categories=cat, handle_unknown='ignore')
        s = oe.fit_transform(np.array(L).reshape(-1, 1)).toarray().transpose(1, 0)
        return(s)


train_chrs = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21', 'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12', 'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
val_chrs = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
#inFs = ['Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_head100.txt', 'Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.txt', 'Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000.txt']
inFs = ['Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_ht100.txt']

if __name__ == '__main__':
    for inF in inFs:
        ouF1 = inF.split('.txt')[0] + '_train.pth'
        ouF2 = inF.split('.txt')[0] + '_val.pth'

        ds_train = CustomDataset(inF, train_chrs)
        ds_val = CustomDataset(inF, val_chrs)

        torch.save(ds_train, ouF1)
        torch.save(ds_val, ouF2)
