import os
import math
import numpy as np
import pandas as pd

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
            ouFile.write('\t'.join([gene, ch, strand, tx_start, tx_end, X, y]) + '\n')

    inFile.close()
    ouFile.close()
