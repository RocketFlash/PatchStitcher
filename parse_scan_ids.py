import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../SEGR.csv', sep=';')
    print(df)
    print(" ".join(df['Scan ID'].dropna().astype('int').astype('str').tolist()))
    pass