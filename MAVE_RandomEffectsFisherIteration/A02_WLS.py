import statsmodels
import statsmodels.api as sm
import pandas as pd

def WLS(inF):
    ouF = inF.split('.txt')[0] + '_Score.txt'
    inFile = open(inF)
    head = inFile.readline().strip().split('\t')
    H = head + ['score', 'SE', 'intercept', 't', 'pvalue']
    Lx = []
    for line in inFile:
        line = line.strip()
        fields = line.split('\t')
        ## X = t/max T
        if inF.find('ReplicatesSumed') != -1:
            y = [float(x) for x in fields[1:4]]
            W = [float(x) for x in fields[4:7]]
            X = [0, 0.5, 1]
        else:
            y = [float(x) for x in fields[1:25]]
            W = [float(x) for x in fields[25:49]]
            X = [0]*8 + [0.5]*8 + [1]*8

        X = sm.add_constant(X)
        fit = sm.WLS(y, X, weights=W).fit()
        L = fields + [fit.params[1], fit.bse[1], fit.params[0], fit.tvalues[1], fit.pvalues[1]]
        Lx.append(L)
    inFile.close()

    df = pd.DataFrame(Lx)
    df.columns = H
    df['padj'] = statsmodels.stats.multitest.multipletests(list(df['pvalue']), method='fdr_bh')[1]
    df.sort_values(by='pvalue', inplace=True)
    df.to_csv(ouF, header=True, index=False, sep='\t')


WLS('MAVE_CodonCounts_InDesignNonConstruct_ReplicatesSumed_LogRatio.txt')
