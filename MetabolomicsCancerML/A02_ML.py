import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_predict
from pyopls import OPLS


inF = 'PlasmaMetabolomics_PCa.txt'
#inF = 'PlasmaMetabolomics_Raw_PCa.txt'
df = pd.read_table(inF, header=0, sep='\t')

y = df['Diagnosis'].values
idx_SampleIDClinical = list(df.columns).index('SAMPLE_ID')
X_meta = df.iloc[:, 1:idx_SampleIDClinical]
X_clin = df.iloc[:, (idx_SampleIDClinical + 1):]
X_meta.index = df.iloc[:, 0]
X_clin.index = df.iloc[:, 0]
X_meta.index.name = None
X_clin.index.name = None
X_clin = X_clin.drop('Diagnosis', axis=1)
print(X_meta.shape)
print(X_clin.shape)

#le = LabelEncoder()
#y = le.fit_transform(y)
y = [1 if x == 1 else -1 for x in y]

'''
inF = 'colorectal_cancer_nmr.csv'
df = pd.read_csv(inF, header=0)
df = df.loc[df['classification'].isin(['Colorectal Cancer', 'Healthy Control']), ]
y = [1 if x == 'Colorectal Cancer' else -1 for x in df['classification']]
#le = LabelEncoder()
#y = le.fit_transform(y)
print(y)
X_meta = df.iloc[:, 1:df.shape[1]-1]
'''

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1]}

def precision_sign(clf, X, y):
    y_pred = clf.predict(X)
    s = precision_score(y, np.sign(y_pred))
    return(s)
def recall_sign(clf, X, y):
    y_pred = clf.predict(X)
    s = recall_score(y, np.sign(y_pred))
    return(s)
def accuracy_sign(clf, X, y):
    y_pred = clf.predict(X)
    s = accuracy_score(y, np.sign(y_pred))
    return(s)

'''
oe = OneHotEncoder()
y = oe.fit_transform(y.reshape(-1, 1)).toarray()
print(y)
'''

############

MODELS = OrderedDict()
#MODELS['LR'] = LogisticRegression()
MODELS['KNN'] = KNeighborsClassifier()
MODELS['GNB'] = GaussianNB()
MODELS['SVM'] = SVC()
MODELS['RF'] = RandomForestClassifier(class_weight='balanced')
#MODELS['XGB'] = XGBClassifier()

def model_train_validate(model, X, y, nF=5, scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc'], confusion=False):
    cv = model_selection.KFold(n_splits=nF, shuffle=True, random_state=0)
    res = model_selection.cross_validate(model, X, y, cv=cv, scoring=scoring)
    L = [round(np.mean(res[f'test_{x}']), 4) for x in scoring]

    if confusion:
        res2 = model_selection.cross_validate(model, X, y, cv=cv, scoring=confusion_matrix_scorer)
        df = pd.DataFrame(np.array([np.sum(res2[f'test_{x}']) for x in ['tp', 'fp', 'fn', 'tn']]).reshape(2, 2))
        df.columns = ['T_real', 'F_real']
        df.index = ['F_pred', 'T_pred']
        print(df)

    return(L)

def feature_engineer(model, X, y, method='kBest', scoring=['precision', 'recall', 'accuracy', 'f1', 'roc_auc'], confusion=False, **kwargs): 
    if method == 'kBest':
        X_new = SelectKBest(f_classif, k=kwargs['kB']).fit_transform(X, y)
        res = model_train_validate(model, X, y, scoring=scoring, confusion=confusion)
        return(res)
    elif method == 'RFE':
        rfe = RFE(model, n_features_to_select=10, step=1).fit(X, y)
        res = model_train_validate(model, X, y, scoring=scoring, confusion=confusion)
        print(res)
    elif method == 'RFECV':
        cv = model_selection.KFold(n_splits=kF, shuffle=True, random_state=0)
        rfecv = RFECV(model, step=1, cv=cv, scoring="accuracy", min_features_to_select=1, n_jobs=2)
    elif method == 'SFM':
        clf = model.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        model_train_validate(model, X, y, scoring=scoring, confusion=confusion)
    elif method == 'PCA':
        model = PCA(kwargs['n_component']).fit(X)
        res = model_train_validate(model, X, y, scoring=scoring, confusion=confusion)
        return(L)
    elif method == 'OPLS':
        X_new = OPLS(kwargs['n_component']).fit_transform(X, y)
        res = model_train_validate(model, X_new, y, scoring=scoring, confusion=confusion)
        return(res)
    elif method == 'None':
        res = model_train_validate(model, X, y, scoring=scoring, confusion=confusion)
        return(res)

def data_split(model, X, y, method='kBest', train_size=0.8, random_state=0, scoring=['precision', 'recall', 'accuracy', 'f1', 'roc_auc'], confusion=False, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state = random_state)
    res = feature_engineer(model, X_train, y_train, method=method, scoring=scoring, confusion=confusion, **kwargs)
    return(res)

'''
for k in MODELS:
    model_name = k
    model = MODELS[k]
    res = data_split(model, X_meta, y, method='kBest', confusion=True, kB=20)
'''


'''
model = MODELS['RF']
for nc in range(20, 100, 10):
    res = data_split(model, X_meta, y, method='OPLS', scoring={'precision_sign':precision_sign, 'recall_sign':recall_sign, 'accuracy_sign':accuracy_sign}, confusion=False, n_component=nc)
    print([nc] + res)
'''


def model_test(model, X, y, train_size=0.8, random_state=0, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state = random_state)
    opls = OPLS(kwargs['n_component']).fit(X_train, y_train)
    X_train_new = opls.transform(X_train)
    X_test_new = opls.transform(X_test)
    model.fit(X_train_new, y_train)
    y_train_pred = model.predict(X_train_new)
    y_test_pred = model.predict(X_test_new)
    print(accuracy_score(y_train, np.sign(y_train_pred)))
    print(accuracy_score(y_test, np.sign(y_test_pred)))

model_test(MODELS['RF'], X_meta, y, n_component=70)

'''
X_train, X_test, y_train, y_test = train_test_split(X_meta, y, train_size=0.8, random_state = 0)
pls = PLSRegression(1)
opls = OPLS(30).fit(X_train, y_train)
X_train_new = opls.transform(X_train)
print(X_train_new.shape)
X_test_new = opls.transform(X_test)
pls.fit(X_train_new, y_train)
y_train_pred = pls.predict(X_train_new)
#y_test_pred = pls.predict(X_test_new)
print(accuracy_score(y_train, np.sign(y_train_pred)))
#print(accuracy_score(y_test, np.sign(y_test_pred)))
'''


'''
X_train, X_test, y_train, y_test = train_test_split(X_meta, y, train_size=0.8, random_state = 0)
for nc in range(1, 10):
    model = PLSRegression(nc)
    res = model_train_validate(model, X_train, y_train, scoring={'accuracy_sign':accuracy_sign})
    print(res)
'''


