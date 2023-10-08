### normalized data
import pandas as pd

def excel2txt(inF1, inF2, ouF):

    df1 = pd.read_excel(inF1, sheet_name=2)
    df1.dropna(axis=1, how='all', inplace=True)

    df2 = pd.read_excel(inF2, sheet_name=1)
    df2.dropna(axis=1, how='all', inplace=True)

    df3 = pd.merge(df1, df2, left_on='CLIENT_SAMPLE_ID', right_on='SAMPLE_ID')
    df3.to_csv(ouF.split('.txt')[0] + '_PCa.txt', header=True, index=False, sep='\t')
    print(df3.loc[:, ['CLIENT_SAMPLE_ID', 'Diagnosis']])

    wh1 = df3['Diagnosis'] == 0
    wh2 = df3['Diagnosis'] == 1
    df4 = df3.loc[wh1, ]
    df5 = df3.loc[wh2, ]
    print(df4)
    print(df5)

    wh1 = df3['Diabetes'] == 0
    wh2 = df3['Diabetes'] == 1
    wh3 = df3['Diagnosis'] == 0
    df4 = df3.loc[wh1 & wh3, ]
    df5 = df3.loc[wh2 & wh3, ]
    df6 = df3.loc[(wh1|wh2) & wh3, ]
    df6.to_csv(ouF.split('.txt')[0] + '_NonPC_Diabetes.txt', header=True, index=False, sep='\t')
    print(df4)
    print(df5)
    print(df6)


excel2txt('230823_EDRN_Data/EDRN_MetabolomicsData_2023_08_27.xlsx', '230823_EDRN_Data/EDRN_ClinicalData.xlsx', ouF='PlasmaMetabolomics.txt')


