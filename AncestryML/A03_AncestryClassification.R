from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import sys

MODELS = [
          ('KNN', KNeighborsClassifier()),
          ('GNB', GaussianNB()),
          ('SVM', SVC()), 
          #('RF', RandomForestClassifier()),
          ('RF', RandomForestClassifier(class_weight='balanced')),
          ('XGB', XGBClassifier()),
        ]

def ML(inF, PCS=range(1, 21), POP='Population'):
    df = pd.read_table(inF, header=0)
    df_sub = df.loc[df['ObsPre'] == 'Observed', :]
    #df_sub2 = df.loc[df['ObsPre'] == 'ToBePredicted', :]
    df_sub2 = df.loc[(df['ObsPre'] == 'ToBePredicted') & ~(df['Sample'].str.startswith('HG') | df['Sample'].str.startswith('NA')), :]
    pIdx = list(df.columns).index(POP)

    le = LabelEncoder()

    X1 = df_sub.iloc[:, PCS]
    y1 = le.fit_transform(df_sub.iloc[:, pIdx])
    X2 = df_sub2.iloc[:, PCS]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state = 0)

    clrdL = []
    for name, model in MODELS:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
        cv_results = model_selection.cross_validate(model, X1_train, y1_train, cv=kfold, scoring=['f1_weighted'])

        clf = model.fit(X1_train, y1_train)
        y1_pred = clf.predict(X1_test)
        print(name)
        clr = classification_report(y1_test, y1_pred)
        print(clr)
        clrd = pd.DataFrame(classification_report(y1_test, y1_pred, output_dict=True))
        print(clrd)
        clrd['model'] = name
        clrd['metrics'] = clrd.index
        clrdL.append(clrd)

        ### SVM doesn't support predict_proba
        if name not in ['SVM']:
            y2_pred = clf.predict(X2)
            y2_prob = clf.predict_proba(X2)

            ouF = inF.split('.txt')[0] + '_%s_%s.txt'%(name, POP)
            DF = pd.DataFrame(y2_prob)
            DF.columns = le.inverse_transform(clf.classes_)
            DF['Sample'] = list(df_sub2['Sample'])
            DF['Class'] = le.inverse_transform(y2_pred)
            DF.to_csv(ouF, header=True, index=False, sep='\t', float_format='%.4f')

    clrdL = pd.concat(clrdL)
    clrdL.to_csv(inF.split('.txt')[0] + '_%s_ClsRreport.txt'%POP, header=True, index=True, sep='\t', float_format='%.4f')
        
 
inF = sys.argv[1]
ML(inF)
ML(inF, POP='Superpopulation')

