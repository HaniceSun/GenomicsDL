### for lib1 only
import pandas as pd
import numpy as np

def LogRatio(inF, SumReplicates=True):

    df = pd.read_table(inF, header=0, sep='\t')

    if SumReplicates:
        df = pd.concat([df.iloc[:, 0:8], df.iloc[:, 8:16].sum(axis=1), df.iloc[:, 16:24].sum(axis=1), df.iloc[:, 24:32].sum(axis=1)], axis=1)
        CL = list(df.columns)
        CL[-3:] = ['Low', 'Middle', 'High']
        df.columns = CL

    ## treat GRCh38 as WT, 152 codons
    #wh = df['GenomeReference'] == 'GRCh38'
    ## treat synonymous as WT, 613 codonds including the 152 codons on GRCh38
    wh = df['REF_AA'] == df['ALT_AA']

    ### output missense only
    #df1 = df.loc[(df['InDesignOrNot'] == 'InDesign') & (~wh), ]
    ### output both missense and synonymous
    df1 = df.loc[(df['InDesignOrNot'] == 'InDesign'), ]
    print(df1.shape)
    df1a = df1.iloc[:, 0:8]
    df1b = df1.iloc[:, 8:]

    df2 = df.loc[(df['InDesignOrNot'] == 'InDesign') & (wh), ]
    print(df2.shape)
    df2a = df2.iloc[:, 0:8]
    df2b = df2.iloc[:, 8:]

    df2b_median = df2b.median(axis=0)
    df2b_mean = df2b.mean(axis=0)
    df2b_sum = df2b.sum(axis=0)

    if not SumReplicates:
        print([df2b_median[0:8].sum(), df2b_median[8:16].sum(), df2b_median[16:24].sum()])
    else:
        pass
        ### for testing, using the three values printed above, to make sure the same output as Enrich2
        #df2b_median = np.array([584.0, 512.0, 689.0])

    df1b_wtMedian = np.log((df1b + 0.5)/(df2b_median + 0.5))
    df1b_wtMedian.columns = [x + '_LogRatio' for x in df1b_wtMedian.columns]
    df1b_wtMedian_weight = 1/(1/(df1b + 0.5) + 1/(df2b_median + 0.5))
    df1b_wtMedian_weight.columns = [x + '_Weight' for x in df1b_wtMedian_weight.columns]

    df1b_RawCounts = df1b
    df1b_RawCounts.columns = [x + '_RawCounts' for x in df1b_RawCounts.columns]

    df1b_RawCountsMean = pd.DataFrame(df1b.mean(axis=1))
    df1b_RawCountsMean.columns = ['RawCountsMean']
    df1b_RawCountsSD = pd.DataFrame(df1b.std(axis=1))
    df1b_RawCountsSD.columns = ['RawCountsSD']

    df1a['Variant'] = df1a['PosAA'].astype(str) + '_' + df1a['REF_Codon'] + '_' + df1a['REF_AA'] + '_' + df1a['ALT_Codon'] + '_' + df1a['ALT_AA']
    df1_wtMedian = pd.concat([df1a['Variant'], df1b_wtMedian, df1b_wtMedian_weight, df1b_RawCounts, df1b_RawCountsMean, df1b_RawCountsSD], axis=1)

    
    if SumReplicates:
        df1_wtMedian.to_csv(inF.split('.txt')[0] + '_ReplicatesSumed_LogRatio.txt', header=True, index=False, sep='\t')
    else:
        df1_wtMedian.to_csv(inF.split('.txt')[0] + '_LogRatio.txt', header=True, index=False, sep='\t')


LogRatio('MAVE_CodonCounts_InDesignNonConstruct.txt')
