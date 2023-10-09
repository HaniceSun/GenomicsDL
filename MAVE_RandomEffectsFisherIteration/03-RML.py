### combine scores of replicates, based on score and SE^2.

### To account for replicate heterogeneity, we use a simple meta-analysis model with a single random effect to combine scores from each of the n replicate selections into a single score for each variant. 
### Each variant’s score is calculated independently. 
### Enrich2 computes the restricted maximum likelihood estimates for the variant score and standard error using Fisher scoring iterations [45]. 
### Given the replicate scores and estimated standard errors, the estimate for 𝛽ˆ at each iteration is the weighted average

import numpy as np
import pandas as pd

def rml_estimator(y, sigma2i, iterations=50):
    """Implementation of the robust maximum likelihood estimator.

        ::

            @book{demidenko2013mixed,
              title={Mixed models: theory and applications with R},
              author={Demidenko, Eugene},
              year={2013},
              publisher={John Wiley \& Sons}
            }

    """
    w = 1 / sigma2i
    sw = np.sum(w, axis=0)
    beta0 = np.sum(y * w, axis=0) / sw
    sigma2ML = np.sum((y - np.mean(y, axis=0)) ** 2 / (len(beta0) - 1), axis=0)
    eps = np.zeros(beta0.shape)
    betaML = None
    for _ in range(iterations):
        w = 1 / (sigma2i + sigma2ML)
        sw = np.sum(w, axis=0)
        sw2 = np.sum(w ** 2, axis=0)
        betaML = np.sum(y * w, axis=0) / sw
        sigma2ML_new = (
            sigma2ML
            * np.sum(((y - betaML) ** 2) * (w ** 2), axis=0)
            / (sw - (sw2 / sw))
        )
        eps = np.abs(sigma2ML - sigma2ML_new)
        sigma2ML = sigma2ML_new
    var_betaML = 1 / np.sum(1 / (sigma2i + sigma2ML), axis=0)
    return betaML, var_betaML, eps


def RME(inF):
    df = pd.read_table(inF, header=0, sep='\t')
    wh1 = [True if x.endswith('_score') else False for x in df.columns]
    wh2 = [True if x.endswith('_SE') else False for x in df.columns]
    score = df.loc[:, wh1]
    SE = df.loc[:, wh2]
    
    y = score.values.T
    sigma2i = np.square(SE).values.T

    betaML, var_betaML, eps = rml_estimator(y, sigma2i)

    df2 = pd.DataFrame({'score':betaML, 'SE':np.sqrt(var_betaML), 'epsilon':eps})

    df3 = pd.concat([df, df2], axis=1)
    df3.sort_values('score', inplace=True)
    df3.to_csv(inF.split('.txt')[0] + '_RME.txt', header=True, index=False, sep='\t')

RME('MAVE_CodonCounts_InDesignNonConstruct_ReplicatesSumed_LogRatio_Score.txt')
