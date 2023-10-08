import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from sklearn.metrics import average_precision_score

def AUPRC(y_true, y_pred):
    auprc = average_precision_score(y_true, y_pred)
    return(auprc)

def TopK(y_true, y_pred, k=1):
    idx_true = np.nonzero(y_true == 1)[0]
    y_pred_argsorted = np.argsort(y_pred)
    y_pred_sorted = np.sort(y_pred)

    th = int(k*len(idx_true))
    idx_pred = y_pred_argsorted[-th:]
    accuracy = len(np.intersect1d(idx_true, idx_pred))/min(len(idx_pred), len(idx_true))
    threshold = y_pred_sorted[-th]
    return([accuracy, threshold])



def MetricsEvaluate(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_NWD1_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val']):
    yTrue = {}
    for ds in DS:
        dataset = tf.data.Dataset.load(inF.split('.tfds')[0] + f'_SMchr-{ds}.tfds', compression='GZIP')
        y_true = np.array([y for x,y in dataset])
        yTrue[ds] = y_true 

    h5s = sorted([x for x in os.listdir(inDir) if x.endswith('ypred.h5')])
    L = []
    for h5 in h5s:
        fields = h5.split('.')
        ckpt = fields[0]
        ds = fields[-3]
        y_true = yTrue[ds]

        f = h5py.File(inDir + '/' + h5, 'r')
        y_pred = np.concatenate([f[k] for k in f.keys()])
        print(y_pred.shape)

        wh = y_true.sum(axis=(1,2)) >= 1
        print(f'{wh.sum()}/{y_pred.shape[0]}')
        yt1 = y_true[wh,:,1].flatten()
        yp1 = y_pred[wh,:,1].flatten()
        yt2 = y_true[wh,:,2].flatten()
        yp2 = y_pred[wh,:,2].flatten()

        acuprc1 = AUPRC(yt1, yp1)
        acuprc2 = AUPRC(yt2, yp2)
        accuracy1, threshold1 = TopK(yt1, yp1)
        accuracy2, threshold2 = TopK(yt2, yp2)

        L.append([ckpt, ds, 'acceptor', auprc1, accuracy1, threshold1])
        L.append([ckpt, ds, 'donor', auprc2, accuracy2, threshold2])

    df = pd.DataFrame(L)
    df.columns = ['CheckPoint', 'Dataset', 'SpliceSite', 'AUPRC', 'TopK', 'Threshold']
    df.to_csv(inDir + '_TopK.txt', sep='\t', header=True, index=False)


MetricsEvaluate(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000_NWD4_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel')
MetricsEvaluate(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank1000.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank1000_NWD3_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel')
MetricsEvaluate(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank200.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank200_NWD2_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel')
MetricsEvaluate(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_NWD1_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel')
