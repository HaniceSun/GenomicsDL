import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def OneHot(L, alphabet=['A', 'C', 'G', 'T']):
    cat = list(np.array(sorted(alphabet)).reshape(1, -1))
    oe = OneHotEncoder(categories=cat, handle_unknown='ignore')
    s = oe.fit_transform(np.array(L).reshape(-1, 1)).toarray()
    return(s)

def getSeqSpliceAI(inF1='canonical_dataset.txt', inF2='hg19.OneLine.fa', trans=''.maketrans('atcgATCG', 'tagcTAGC')):
    FA = {}
    inFile = open(inF2)
    while True:
        line1 = inFile.readline().strip()
        line2 = inFile.readline().strip()
        if line1:
            FA[line1[1:]] = '.' + line2.upper()
        else:
            break
    inFile.close()

    ouFile = open(inF1.split('.txt')[0] + '_seq.txt', 'w')
    inFile = open(inF1)
    for line in inFile:
        line = line.strip()
        fields = line.split('\t')
        gene = fields[0]
        ch = fields[2]
        strand = fields[3]
        tx_start = int(fields[4])
        tx_end = int(fields[5])
        exon_start = [int(x) for x in fields[6].split(',')[0:-1]]
        exon_end = [int(x) for x in fields[7].split(',')[0:-1]]
        if ch in FA:
            if strand == '+':
                tx_seq = FA[ch][tx_start:tx_end+1]
                tx_seq_y = [0] * len(tx_seq)
                for x in exon_start:
                    tx_seq_y[x - tx_start] = 2
                for x in exon_end:
                    tx_seq_y[x - tx_start] = 1
            if strand == '-':
                tx_seq = FA[ch][tx_start:tx_end+1]
                tx_seq_y = [0] * len(tx_seq)
                for x in exon_start:
                    tx_seq_y[x - tx_start] = 1
                for x in exon_end:
                    tx_seq_y[x - tx_start] = 2
                tx_seq = tx_seq[::-1].translate(trans)
                tx_seq_y = tx_seq_y[::-1]
            tx_seq_y = ''.join([str(x) for x in tx_seq_y])
            ouFile.write(line + '\t' + tx_seq + '\t' + tx_seq_y + '\n')
    inFile.close()
    ouFile.close()

def getSeqBED12(inF1='Homo_sapiens.GRCh38.110.bed12', inF2='/oak/stanford/groups/agloyn/hansun/Data/Ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.OneLine.fa', trans=''.maketrans('atcgATCG', 'tagcTAGC'), CH=['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY'], extra=['Ensembl_canonical', 'protein_coding']):
    FA = {}
    inFile = open(inF2)
    while True:
        line1 = inFile.readline().strip()
        line2 = inFile.readline().strip()
        if line1:
            ch = 'chr' + line1.split()[0][1:]
            FA[ch] = '.' + line2.upper()
        else:
            break
    inFile.close()

    ouFile = open(inF1.split('/')[-1].split('.txt')[0] + '_seq.txt', 'w')
    inFile = open(inF1)
    for line in inFile:
        line = line.strip()
        fields = line.split('\t')
        gene = fields[12] + '_' + fields[13]
        ch = 'chr' + fields[0]
        strand = fields[5]
        tx_start = int(fields[1]) + 1
        tx_end = int(fields[2])
        exon_start = []
        exon_end = []
        blockStarts = fields[11].split(',')[0:-1]
        blockSizes = fields[10].split(',')[0:-1]
        for n in range(len(blockStarts)):
            exon_start.append(int(fields[1]) + 1 + int(blockStarts[n]))
            exon_end.append(int(fields[1]) + int(blockStarts[n]) + int(blockSizes[n]))

        flag = True
        if fields[18].find(extra[0]) == -1 or fields[14].find(extra[1]) == -1:
                flag = False
        if flag and ch in FA and len(exon_start) > 1 and ch in CH:
            if strand == '+':
                # donor:2, acceptor:1
                tx_seq = FA[ch][tx_start:tx_end+1]
                tx_seq_y = [0] * len(tx_seq)
                for x in exon_start[1:]:
                    tx_seq_y[x - tx_start] = 1
                for x in exon_end[0:-1]:
                    tx_seq_y[x - tx_start] = 2
            if strand == '-':
                tx_seq = FA[ch][tx_start:tx_end+1]
                tx_seq_y = [0] * len(tx_seq)
                for x in exon_start[1:]:
                    tx_seq_y[x - tx_start] = 2
                for x in exon_end[0:-1]:
                    tx_seq_y[x - tx_start] = 1
                tx_seq = tx_seq[::-1].translate(trans)
                tx_seq_y = tx_seq_y[::-1]
            tx_seq_y = ''.join([str(x) for x in tx_seq_y])
            ouFile.write('chr' + line + '\t' + ','.join([str(x) for x in exon_start]) + '\t' + ','.join([str(x) for x in exon_end]) + '\t' + tx_seq + '\t' + tx_seq_y + '\n')
    inFile.close()
    ouFile.close()


def splitSeq(inF='canonical_dataset_seq.txt', nt=5000, FlankSize=5000):
    inFile = open(inF)
    ouF = inF.split('.txt')[0] + f"_nt{nt}_flank{FlankSize}.txt"
    ouFile = open(ouF, 'w')
    ouFile.write('\t'.join(['gene', 'ch', 'strand', 'start', 'end', 'X', 'y']) + '\n')
    for line in inFile:
        line = line.strip()
        fields = line.split('\t')
        if inF.find('canonical_dataset') == 0:
            gene = fields[0]
            ch = fields[2]
            strand = fields[3]
            tx_start = fields[4]
            tx_end = fields[5]
            tx_seq = fields[8]
            tx_seq_y = fields[9]
        elif inF.find('bed12') != -1:
            gene = fields[12] + '|' + fields[13]
            ch = fields[0]
            strand = fields[5]
            tx_start = str(int(fields[1]) + 1)
            tx_end = fields[2]
            tx_seq = fields[-2]
            tx_seq_y = fields[-1]

        tx_seq_pad = tx_seq + 'N' * (math.ceil(len(tx_seq)/nt)*nt-len(tx_seq))
        tx_seq_y_pad = tx_seq_y + 'N' * (math.ceil(len(tx_seq_y)/nt)*nt-len(tx_seq_y))
        for n in range(0, len(tx_seq_pad), nt):
            start = n-FlankSize
            end = n + nt + FlankSize
            left = ''
            right = ''
            if start < 0:
                left = 'N' * abs(start)
                start = 0
            if end > len(tx_seq_pad):
                right = 'N' * (end - len(tx_seq_pad))
                end = len(tx_seq_pad)
            X = left + tx_seq_pad[start:end] + right
            y = tx_seq_y_pad[n:n+nt]
            ##X = X.replace('A', '1').replace('T', '4').replace('C', '2').replace('G', '3').replace('N', '0')
            ##y = y.replace('N', '-1')
            ouFile.write('\t'.join([gene, ch, strand, tx_start, tx_end, X, y]) + '\n')

    inFile.close()
    ouFile.close()

    getDataset(ouF)

def getDataset(inF='canonical_dataset_seq_nt5000_flank5000.txt'):
    ouF = inF.split('.txt')[0] + '.tfds'
    df = pd.read_table(inF, header=0, sep='\t')
    X_ds = tf.data.Dataset.from_tensor_slices([OneHot(list(x)) for x in df['X']])
    y_ds =tf.data.Dataset.from_tensor_slices([OneHot(list(x), alphabet=['0', '1', '2']) for x in df['y']])
    extra_ds = tf.data.Dataset.from_tensor_slices(df.iloc[:, 0:-2].apply(lambda x: '_'.join(x.astype(str)), axis=1).values.reshape(-1, 1))
    dataset = tf.data.Dataset.zip((X_ds, y_ds, extra_ds))
    dataset = dataset.shuffle(buffer_size=df.shape[0])
    tf.data.Dataset.save(dataset, ouF, compression='GZIP')

    splitDataset(ouF, splitPara=['chr', train_chrs, val_chrs, test_chrs])


def splitDataset(inF, splitPara=['percent', 0.8, 0.1, 0.1]):
    dataset = tf.data.Dataset.load(inF, compression='GZIP')
    if splitPara[0] in ['percent']:
        dataset = dataset.map(lambda x,y,z:(x, y))
        nSample = len(dataset)
        nTrain = int(nSample * splitPara[1])
        nVal = int(nSample * splitPara[2])
        nTest = nSample - nTrain - nVal
        print([nTrain, nVal, nTest])
        dataset_train = dataset.take(nTrain)
        dataset_val = dataset.skip(nTrain).take(nVal)
        dataset_test = dataset.skip(nTrain + nVal).take(nTest)
    elif splitPara[0] in ['chr']:
        dataset_train = tf.data.experimental.from_list([x[0:2] for x in dataset if x[2].numpy()[0].decode().split('_')[1] in splitPara[1]])
        dataset_val = tf.data.experimental.from_list([x[0:2] for x in dataset if x[2].numpy()[0].decode().split('_')[1] in splitPara[2]])
        dataset_test = tf.data.experimental.from_list([x[0:2] for x in dataset if x[2].numpy()[0].decode().split('_')[1] in splitPara[3]])
        print([len(dataset_train), len(dataset_val), len(dataset_test)])

    tf.data.Dataset.save(dataset_train, inF.split('.tfds')[0] + '_SM%s-train.tfds'%splitPara[0], compression='GZIP')
    tf.data.Dataset.save(dataset_val, inF.split('.tfds')[0] + '_SM%s-val.tfds'%splitPara[0], compression='GZIP')
    tf.data.Dataset.save(dataset_test, inF.split('.tfds')[0] + '_SM%s-test.tfds'%splitPara[0], compression='GZIP')


train_chrs = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21', 'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12', 'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
val_chrs = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
# same as val temporarily
test_chrs = val_chrs

getSeqBED12('Homo_sapiens.GRCh38.110.bed12')
splitSeq(inF='Homo_sapiens.GRCh38.110.bed12_seq.txt', FlankSize=40)
getDataset('Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.txt')
splitDataset('Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.tfds', splitPara=['chr', train_chrs, val_chrs, test_chrs])
